//! GPU kernel source code for different compute backends
//!
//! This module contains kernel implementations for CUDA, OpenCL, Metal, and Vulkan
//! optimized for metrics computation.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

/// CUDA kernel source code for metrics computation
pub mod cuda_kernels {
    pub const MSE_KERNEL: &str = r#"
    extern "C" __global__ void mse_kernel(
        const float* y_true,
        const float* ypred,
        float* result,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        __shared__ float sdata[256];

        float sum = 0.0f;
        for (int i = idx; i < n; i += stride) {
            float diff = y_true[i] - ypred[i];
            sum += diff * diff;
        }

        sdata[threadIdx.x] = sum;
        __syncthreads();

        // Parallel reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(result, sdata[0] / n);
        }
    }
    "#;

    pub const MAE_KERNEL: &str = r#"
    extern "C" __global__ void mae_kernel(
        const float* y_true,
        const float* ypred,
        float* result,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        __shared__ float sdata[256];

        float sum = 0.0f;
        for (int i = idx; i < n; i += stride) {
            sum += fabsf(y_true[i] - ypred[i]);
        }

        sdata[threadIdx.x] = sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(result, sdata[0] / n);
        }
    }
    "#;

    pub const R2_KERNEL: &str = r#"
    extern "C" __global__ void r2_kernel(
        const float* y_true,
        const float* ypred,
        float* ss_res,
        float* ss_tot,
        float mean_true,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        __shared__ float sdata_res[256];
        __shared__ float sdata_tot[256];

        float sum_res = 0.0f;
        float sum_tot = 0.0f;

        for (int i = idx; i < n; i += stride) {
            float diff = y_true[i] - ypred[i];
            sum_res += diff * diff;

            float diff_mean = y_true[i] - mean_true;
            sum_tot += diff_mean * diff_mean;
        }

        sdata_res[threadIdx.x] = sum_res;
        sdata_tot[threadIdx.x] = sum_tot;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_res[threadIdx.x] += sdata_res[threadIdx.x + s];
                sdata_tot[threadIdx.x] += sdata_tot[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(ss_res, sdata_res[0]);
            atomicAdd(ss_tot, sdata_tot[0]);
        }
    }
    "#;

    pub const PRECISION_RECALL_KERNEL: &str = r#"
    extern "C" __global__ void precision_recall_kernel(
        const float* y_true,
        const float* ypred,
        float* tp,
        float* fp,
        float* fn_ptr,
        float threshold,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        __shared__ float sdata_tp[256];
        __shared__ float sdata_fp[256];
        __shared__ float sdata_fn[256];

        float local_tp = 0.0f;
        float local_fp = 0.0f;
        float local_fn = 0.0f;

        for (int i = idx; i < n; i += stride) {
            float pred = ypred[i] > threshold ? 1.0f : 0.0f;
            float truth = y_true[i];

            if (pred == 1.0f && truth == 1.0f) local_tp += 1.0f;
            else if (pred == 1.0f && truth == 0.0f) local_fp += 1.0f;
            else if (pred == 0.0f && truth == 1.0f) local_fn += 1.0f;
        }

        sdata_tp[threadIdx.x] = local_tp;
        sdata_fp[threadIdx.x] = local_fp;
        sdata_fn[threadIdx.x] = local_fn;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata_tp[threadIdx.x] += sdata_tp[threadIdx.x + s];
                sdata_fp[threadIdx.x] += sdata_fp[threadIdx.x + s];
                sdata_fn[threadIdx.x] += sdata_fn[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(tp, sdata_tp[0]);
            atomicAdd(fp, sdata_fp[0]);
            atomicAdd(fn_ptr, sdata_fn[0]);
        }
    }
    "#;
}

/// OpenCL kernel source code for metrics computation
pub mod opencl_kernels {
    pub const MSE_KERNEL: &str = r#"
    __kernel void mse_kernel(
        __global const float* y_true,
        __global const float* ypred,
        __global float* result,
        const int n
    ) {
        int idx = get_global_id(0);
        int stride = get_global_size(0);

        __local float sdata[256];
        int lid = get_local_id(0);

        float sum = 0.0f;
        for (int i = idx; i < n; i += stride) {
            float diff = y_true[i] - ypred[i];
            sum += diff * diff;
        }

        sdata[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            atomic_add_global(result, sdata[0] / n);
        }
    }
    "#;

    pub const MAE_KERNEL: &str = r#"
    __kernel void mae_kernel(
        __global const float* y_true,
        __global const float* ypred,
        __global float* result,
        const int n
    ) {
        int idx = get_global_id(0);
        int stride = get_global_size(0);

        __local float sdata[256];
        int lid = get_local_id(0);

        float sum = 0.0f;
        for (int i = idx; i < n; i += stride) {
            sum += fabs(y_true[i] - ypred[i]);
        }

        sdata[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            atomic_add_global(result, sdata[0] / n);
        }
    }
    "#;
}

/// Metal compute shader kernels for metrics computation
pub mod metal_kernels {
    pub const MSE_KERNEL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void mse_kernel(
        device const float* y_true [[buffer(0)]],
        device const float* ypred [[buffer(1)]],
        device float* result [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint id [[thread_position_in_grid]],
        uint threads_per_grid [[threads_per_grid]]
    ) {
        threadgroup float sdata[256];
        uint lid = threadgroup_position_in_grid;

        float sum = 0.0;
        for (uint i = id; i < n; i += threads_per_grid) {
            float diff = y_true[i] - ypred[i];
            sum += diff * diff;
        }

        sdata[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 128; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            atomic_fetch_add_explicit(result, sdata[0] / n, memory_order_relaxed);
        }
    }
    "#;

    pub const MAE_KERNEL: &str = r#"
    #include <metal_stdlib>
    using namespace metal;

    kernel void mae_kernel(
        device const float* y_true [[buffer(0)]],
        device const float* ypred [[buffer(1)]],
        device float* result [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint id [[thread_position_in_grid]],
        uint threads_per_grid [[threads_per_grid]]
    ) {
        threadgroup float sdata[256];
        uint lid = threadgroup_position_in_grid;

        float sum = 0.0;
        for (uint i = id; i < n; i += threads_per_grid) {
            sum += abs(y_true[i] - ypred[i]);
        }

        sdata[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 128; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            atomic_fetch_add_explicit(result, sdata[0] / n, memory_order_relaxed);
        }
    }
    "#;
}

/// Vulkan SPIR-V compute shader kernels
pub mod vulkan_kernels {
    pub const MSE_SPIRV: &[u8] = &[
        // SPIR-V bytecode for MSE kernel would go here
        // This is a placeholder for the actual compiled SPIR-V
        0x03, 0x02, 0x23,
        0x07, // SPIR-V magic number
             // ... actual SPIR-V bytecode would follow
    ];

    pub const MAE_SPIRV: &[u8] = &[
        // SPIR-V bytecode for MAE kernel would go here
        0x03, 0x02, 0x23,
        0x07, // SPIR-V magic number
             // ... actual SPIR-V bytecode would follow
    ];

    pub const MSE_GLSL_SOURCE: &str = r#"
    #version 450

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    layout(set = 0, binding = 0) buffer YTrue {
        float y_true[];
    };

    layout(set = 0, binding = 1) buffer YPred {
        float ypred[];
    };

    layout(set = 0, binding = 2) buffer Result {
        float result[];
    };

    layout(push_constant) uniform PushConstants {
        uint n;
    } pc;

    shared float sdata[256];

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
        uint lid = gl_LocalInvocationID.x;

        float sum = 0.0;
        for (uint i = idx; i < pc.n; i += stride) {
            float diff = y_true[i] - ypred[i];
            sum += diff * diff;
        }

        sdata[lid] = sum;
        barrier();

        for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            barrier();
        }

        if (lid == 0) {
            atomicAdd(result[0], sdata[0] / pc.n);
        }
    }
    "#;

    pub const MAE_GLSL_SOURCE: &str = r#"
    #version 450

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

    layout(set = 0, binding = 0) buffer YTrue {
        float y_true[];
    };

    layout(set = 0, binding = 1) buffer YPred {
        float ypred[];
    };

    layout(set = 0, binding = 2) buffer Result {
        float result[];
    };

    layout(push_constant) uniform PushConstants {
        uint n;
    } pc;

    shared float sdata[256];

    void main() {
        uint idx = gl_GlobalInvocationID.x;
        uint stride = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
        uint lid = gl_LocalInvocationID.x;

        float sum = 0.0;
        for (uint i = idx; i < pc.n; i += stride) {
            sum += abs(y_true[i] - ypred[i]);
        }

        sdata[lid] = sum;
        barrier();

        for (uint s = gl_WorkGroupSize.x / 2; s > 0; s >>= 1) {
            if (lid < s) {
                sdata[lid] += sdata[lid + s];
            }
            barrier();
        }

        if (lid == 0) {
            atomicAdd(result[0], sdata[0] / pc.n);
        }
    }
    "#;
}
