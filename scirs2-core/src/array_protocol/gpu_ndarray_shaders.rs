// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! WGSL kernel sources for `GpuNdarray` operations.
//!
//! Split from `gpu_ndarray.rs` to keep each file under 2000 lines.

/// Elementwise add: result[i] = a[i] + b[i].
/// Uses arrayLength to avoid uniform binding.
pub(super) const ELEMENTWISE_ADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] + b[idx];
}
"#;

/// Elementwise subtract: result[i] = a[i] - b[i].
pub(super) const ELEMENTWISE_SUB_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] - b[idx];
}
"#;

/// Elementwise multiply: result[i] = a[i] * b[i].
pub(super) const ELEMENTWISE_MUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&result)) { return; }
    result[idx] = a[idx] * b[idx];
}
"#;

/// Elementwise scalar multiply: result[i] = a[i] * scalar.
pub(super) const SCALAR_MUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read_write> result : array<f32>;

struct Uniforms {
    scalar : f32,
    n      : u32,
};
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.n) { return; }
    result[idx] = a[idx] * uniforms.scalar;
}
"#;

/// Naive matmul C = A × B (one thread per output element).
///
/// uniforms: M, N, K (dimensions of A[M,K] and B[K,N] → C[M,N]).
/// Workgroup (16,16,1) → each invocation computes one C[row,col].
pub(super) const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a_mat : array<f32>;
@group(0) @binding(1) var<storage, read>       b_mat : array<f32>;
@group(0) @binding(2) var<storage, read_write> c_mat : array<f32>;

struct Uniforms {
    M : u32,
    N : u32,
    K : u32,
    _pad : u32,
};
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= uniforms.M || col >= uniforms.N) { return; }

    var acc : f32 = 0.0;
    for (var k : u32 = 0u; k < uniforms.K; k++) {
        acc += a_mat[row * uniforms.K + k] * b_mat[k * uniforms.N + col];
    }
    c_mat[row * uniforms.N + col] = acc;
}
"#;

/// Two-pass sum reduction — workgroup partial sums.
///
/// uniforms: n (total element count).
/// partial[workgroup_id] = sum of up to 256 elements in that workgroup.
pub(super) const SUM_REDUCE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input   : array<f32>;
@group(0) @binding(1) var<storage, read_write> partial : array<f32>;

struct Uniforms {
    n : u32,
};
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

var<workgroup> wg_data : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>
) {
    let idx = gid.x;
    let local_idx = lid.x;

    if (idx < uniforms.n) {
        wg_data[local_idx] = input[idx];
    } else {
        wg_data[local_idx] = 0.0;
    }
    workgroupBarrier();

    var stride : u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (local_idx < stride) {
            wg_data[local_idx] += wg_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        partial[wgid.x] = wg_data[0];
    }
}
"#;

/// Bank-conflict-padded 16×16 2-D transpose.
///
/// uniforms: rows (u32), cols (u32) of input.
/// output shape is [cols, rows].
/// Uses 16×16 workgroup (256 total) to fit within Metal's 256 invocation limit.
pub(super) const TRANSPOSE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Uniforms {
    rows : u32,
    cols : u32,
};
@group(0) @binding(2) var<uniform> uniforms : Uniforms;

// +1 pad avoids bank conflicts for 16-wide tiles
var<workgroup> tile : array<f32, 272>;  // 16 * 17

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wgid : vec3<u32>
) {
    let in_col  = wgid.x * 16u + lid.x;
    let in_row  = wgid.y * 16u + lid.y;

    if (in_row < uniforms.rows && in_col < uniforms.cols) {
        tile[lid.y * 17u + lid.x] = input[in_row * uniforms.cols + in_col];
    }
    workgroupBarrier();

    let out_col = wgid.y * 16u + lid.x;
    let out_row = wgid.x * 16u + lid.y;
    if (out_row < uniforms.cols && out_col < uniforms.rows) {
        output[out_row * uniforms.rows + out_col] = tile[lid.x * 17u + lid.y];
    }
}
"#;

/// Concatenate two tensors along an arbitrary axis.
///
/// Bindings:
///   0 = a (storage, read)     — first input, flat f32
///   1 = b (storage, read)     — second input, flat f32
///   2 = result (storage, rw)  — output, flat f32
///   3 = uniforms (uniform)    — ConcatUniforms: axis, dim_a, ndim, _pad
///   4 = out_shape  (storage, read) — u32 array, length = ndim
///   5 = out_strides (storage, read) — u32 array, length = ndim
///   6 = a_strides  (storage, read) — u32 array, length = ndim
///   7 = b_strides  (storage, read) — u32 array, length = ndim
///
/// Shape/strides are storage buffers to avoid WGSL uniform 16-byte-per-element alignment.
pub(super) const CONCAT_AXISN_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a_buf       : array<f32>;
@group(0) @binding(1) var<storage, read>       b_buf       : array<f32>;
@group(0) @binding(2) var<storage, read_write> result      : array<f32>;

struct ConcatUniforms {
    axis  : u32,
    dim_a : u32,
    ndim  : u32,
    _pad  : u32,
};
@group(0) @binding(3) var<uniform>             uniforms    : ConcatUniforms;

@group(0) @binding(4) var<storage, read>       out_shape   : array<u32>;
@group(0) @binding(5) var<storage, read>       out_strides : array<u32>;
@group(0) @binding(6) var<storage, read>       a_strides   : array<u32>;
@group(0) @binding(7) var<storage, read>       b_strides   : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    let n_out = arrayLength(&result);
    if (out_idx >= n_out) { return; }

    let ndim = uniforms.ndim;
    let axis = uniforms.axis;
    let dim_a = uniforms.dim_a;

    // Decompose out_idx into multi-dimensional coordinates using out_strides
    var remaining : u32 = out_idx;
    var coords : array<u32, 8>;
    for (var d : u32 = 0u; d < ndim; d++) {
        let s = out_strides[d];
        coords[d] = remaining / s;
        remaining  = remaining % s;
    }

    // Determine whether this coordinate comes from A or B
    let ax_coord = coords[axis];
    if (ax_coord < dim_a) {
        // Read from A: compute flat index using a_strides
        var a_idx : u32 = 0u;
        for (var d : u32 = 0u; d < ndim; d++) {
            a_idx += coords[d] * a_strides[d];
        }
        result[out_idx] = a_buf[a_idx];
    } else {
        // Read from B: shift axis coordinate by -dim_a
        coords[axis] = ax_coord - dim_a;
        var b_idx : u32 = 0u;
        for (var d : u32 = 0u; d < ndim; d++) {
            b_idx += coords[d] * b_strides[d];
        }
        result[out_idx] = b_buf[b_idx];
    }
}
"#;

/// Reduce one axis of an N-D tensor via summation.
///
/// Bindings:
///   0 = input  (storage, read)    — flat f32, shape in_shape
///   1 = result (storage, rw)      — flat f32, shape out_shape (axis removed)
///   2 = uniforms (uniform)        — ReduceUniforms: axis, axis_size, ndim, in_axis_stride
///   3 = in_shape   (storage, read) — u32 array, length = ndim
///   4 = in_strides (storage, read) — u32 array, length = ndim
///   5 = out_shape  (storage, read) — u32 array, length = ndim-1
///   6 = out_strides (storage, read) — u32 array, length = ndim-1
///
/// Each invocation handles one output element.
pub(super) const REDUCE_SUM_AXIS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       input       : array<f32>;
@group(0) @binding(1) var<storage, read_write> result      : array<f32>;

struct ReduceUniforms {
    axis           : u32,
    axis_size      : u32,
    ndim           : u32,
    in_axis_stride : u32,
};
@group(0) @binding(2) var<uniform>             uniforms    : ReduceUniforms;

@group(0) @binding(3) var<storage, read>       in_shape    : array<u32>;
@group(0) @binding(4) var<storage, read>       in_strides  : array<u32>;
@group(0) @binding(5) var<storage, read>       out_shape   : array<u32>;
@group(0) @binding(6) var<storage, read>       out_strides : array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_idx = gid.x;
    let n_out = arrayLength(&result);
    if (out_idx >= n_out) { return; }

    let ndim = uniforms.ndim;
    let axis = uniforms.axis;
    let axis_size = uniforms.axis_size;
    let in_axis_stride = uniforms.in_axis_stride;
    let out_ndim = ndim - 1u;

    // Decompose out_idx into coordinates for the (ndim-1)-dim output
    var remaining : u32 = out_idx;
    var out_coords : array<u32, 8>;
    for (var d : u32 = 0u; d < out_ndim; d++) {
        let s = out_strides[d];
        out_coords[d] = remaining / s;
        remaining      = remaining % s;
    }

    // Map out_coords back to input coordinates (insert 0 for the reduction axis)
    // and compute base flat offset in input with axis index = 0
    var base_in : u32 = 0u;
    var od : u32 = 0u;  // out dimension cursor
    for (var d : u32 = 0u; d < ndim; d++) {
        if (d == axis) {
            // reduction axis: contribute 0 (will be summed over j below)
        } else {
            base_in += out_coords[od] * in_strides[d];
            od += 1u;
        }
    }

    // Sum over all elements along the reduction axis
    var acc : f32 = 0.0;
    for (var j : u32 = 0u; j < axis_size; j++) {
        acc += input[base_in + j * in_axis_stride];
    }
    result[out_idx] = acc;
}
"#;
