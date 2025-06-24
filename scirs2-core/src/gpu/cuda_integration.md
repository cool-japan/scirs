# CUDA Integration Implementation Guide for scirs2-core

## Recommended Approach: Using High-Level Crates

### 1. Update Cargo.toml

```toml
[dependencies.cudarc]
version = "0.10"
optional = true
features = ["cudnn", "cublas", "cufft"]

[dependencies.candle]
version = "0.3"
optional = true
features = ["cuda"]

[features]
cuda = ["dep:cudarc", "dep:candle", "gpu"]
```

### 2. Create CUDA Backend Module

```rust
// src/gpu/backends/cuda_impl.rs
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::cublas::{CudaBlas, Gemm};
use crate::gpu::{GpuError, GpuBuffer, GpuContext};

pub struct CudaBackend {
    device: Arc<CudaDevice>,
    blas: Arc<CudaBlas>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0)
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
        
        let blas = CudaBlas::new(device.clone())
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
        
        Ok(Self {
            device: Arc::new(device),
            blas: Arc::new(blas),
        })
    }
    
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        unsafe {
            self.blas.gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m as i32, n as i32, k as i32,
                &alpha,
                a, m as i32,
                b, k as i32,
                &beta,
                c, m as i32,
            ).map_err(|e| GpuError::KernelExecutionFailed(e.to_string()))?;
        }
        Ok(())
    }
}
```

### 3. Implement GPU Buffer

```rust
// src/gpu/buffer_cuda.rs
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};
use std::sync::Arc;

pub struct CudaBuffer<T: DeviceRepr> {
    device: Arc<CudaDevice>,
    data: CudaSlice<T>,
    size: usize,
}

impl<T: DeviceRepr + Default + Clone> CudaBuffer<T> {
    pub fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self, GpuError> {
        let data = device.alloc_zeros(size)
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;
        
        Ok(Self { device, data, size })
    }
    
    pub fn from_slice(device: Arc<CudaDevice>, slice: &[T]) -> Result<Self, GpuError> {
        let data = device.htod_sync_copy(slice)
            .map_err(|e| GpuError::TransferFailed(e.to_string()))?;
        
        Ok(Self {
            device,
            size: slice.len(),
            data,
        })
    }
    
    pub fn to_vec(&self) -> Result<Vec<T>, GpuError> {
        self.device.dtoh_sync_copy(&self.data)
            .map_err(|e| GpuError::TransferFailed(e.to_string()))
    }
}
```

### 4. Custom CUDA Kernels

```rust
// src/gpu/kernels/custom_cuda.rs
use cudarc::driver::{CudaDevice, CudaFunction, CudaModule};

const VECTOR_OPS_PTX: &str = r#"
.version 7.0
.target sm_50
.address_size 64

.visible .entry vector_add_f32(
    .param .u64 vector_add_f32_param_0,
    .param .u64 vector_add_f32_param_1,
    .param .u64 vector_add_f32_param_2,
    .param .u32 vector_add_f32_param_3
)
{
    .reg .pred      %p<2>;
    .reg .f32       %f<4>;
    .reg .b32       %r<6>;
    .reg .b64       %rd<10>;

    ld.param.u64    %rd1, [vector_add_f32_param_0];
    ld.param.u64    %rd2, [vector_add_f32_param_1];
    ld.param.u64    %rd3, [vector_add_f32_param_2];
    ld.param.u32    %r2, [vector_add_f32_param_3];
    
    mov.u32         %r3, %ntid.x;
    mov.u32         %r4, %ctaid.x;
    mov.u32         %r5, %tid.x;
    mad.lo.s32      %r1, %r4, %r3, %r5;
    
    setp.ge.s32     %p1, %r1, %r2;
    @%p1 bra        LBB0_2;
    
    cvt.s64.s32     %rd4, %r1;
    mul.wide.s32    %rd5, %r1, 4;
    add.s64         %rd6, %rd2, %rd5;
    add.s64         %rd7, %rd1, %rd5;
    ld.global.f32   %f1, [%rd7];
    ld.global.f32   %f2, [%rd6];
    add.f32         %f3, %f1, %f2;
    add.s64         %rd8, %rd3, %rd5;
    st.global.f32   [%rd8], %f3;
    
LBB0_2:
    ret;
}
"#;

pub struct CudaKernels {
    device: Arc<CudaDevice>,
    module: CudaModule,
}

impl CudaKernels {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, GpuError> {
        let module = device.load_ptx(VECTOR_OPS_PTX.into(), "vector_ops", &[])
            .map_err(|e| GpuError::InitializationFailed(e.to_string()))?;
        
        Ok(Self { device, module })
    }
    
    pub fn vector_add_f32(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self.module.get_function("vector_add_f32")
            .map_err(|e| GpuError::KernelNotFound(e.to_string()))?;
        
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            f.launch(config, (a, b, c, n as u32))
                .map_err(|e| GpuError::KernelExecutionFailed(e.to_string()))?;
        }
        
        Ok(())
    }
}
```

### 5. Integration Example

```rust
// examples/cuda_integration_test.rs
use scirs2_core::gpu::{GpuContext, GpuBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize CUDA backend
    let ctx = GpuContext::new(GpuBackend::Cuda)?;
    
    // Large-scale computation
    let n = 10_000_000; // 10M elements
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.002).collect();
    
    // Upload to GPU
    let gpu_a = ctx.create_buffer_from_slice(&a)?;
    let gpu_b = ctx.create_buffer_from_slice(&b)?;
    let mut gpu_c = ctx.create_buffer::<f32>(n)?;
    
    // Execute on GPU
    println!("Executing vector addition on GPU...");
    let start = std::time::Instant::now();
    
    ctx.vector_add(&gpu_a, &gpu_b, &mut gpu_c)?;
    ctx.synchronize()?;
    
    let elapsed = start.elapsed();
    println!("GPU computation completed in {:?}", elapsed);
    
    // Get results
    let c = gpu_c.to_vec()?;
    
    // Verify
    for i in 0..5 {
        println!("c[{}] = {} (expected {})", i, c[i], a[i] + b[i]);
    }
    
    Ok(())
}
```

## Benefits of This Approach

1. **Type Safety**: Rust's type system ensures memory safety
2. **Automatic Memory Management**: RAII pattern handles cleanup
3. **Error Handling**: Proper error propagation
4. **High Performance**: Direct CUDA API access
5. **Ecosystem Integration**: Works with existing Rust libraries

## Next Steps

1. Add this to scirs2-core as an optional feature
2. Implement CUDA kernels for all major operations
3. Add benchmarks comparing CPU vs GPU performance
4. Create comprehensive tests
5. Document API usage