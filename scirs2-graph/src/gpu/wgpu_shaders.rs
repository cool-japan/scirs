//! WGSL compute shader sources for GPU graph algorithms.
//!
//! This module is compiled only when the `gpu` feature is enabled.
//! Each shader constant contains a self-contained WGSL compute program.

/// Level-synchronous BFS frontier expansion shader.
///
/// Bindings (group 0):
/// - 0: `row_offsets` — CSR row offset array (read-only, `array<u32>`)
/// - 1: `col_indices` — CSR column index array (read-only, `array<u32>`)
/// - 2: `distances`   — output distance array, init to i32::MAX=-1 (read-write, `array<atomic<i32>>`)
/// - 3: `frontier_in` — current frontier vertex list (read-only, `array<u32>`)
/// - 4: `frontier_out` — next frontier vertex list (read-write, `array<atomic<u32>>`)
/// - 5: `frontier_out_count` — next frontier vertex count (read-write, `atomic<u32>`)
/// - 6: `uniforms` — `{ frontier_size: u32, current_level: i32 }` (uniform)
///
/// Each invocation handles one vertex in `frontier_in`. It iterates over the
/// neighbors of that vertex and tries `atomicCompareExchange` on `distances[neighbor]`
/// from -1 (unvisited) to `current_level + 1`.  Successful CAS → append to `frontier_out`.
pub const BFS_FRONTIER_WGSL: &str = r#"
struct BfsUniforms {
    frontier_size: u32,
    current_level: i32,
};

@group(0) @binding(0) var<storage, read>       row_offsets        : array<u32>;
@group(0) @binding(1) var<storage, read>       col_indices        : array<u32>;
@group(0) @binding(2) var<storage, read_write> distances          : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read>       frontier_in        : array<u32>;
@group(0) @binding(4) var<storage, read_write> frontier_out       : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> frontier_out_count : atomic<u32>;
@group(0) @binding(6) var<uniform>             uniforms           : BfsUniforms;

@compute @workgroup_size(64)
fn bfs_frontier(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.frontier_size) {
        return;
    }
    let v = frontier_in[tid];
    let row_start = row_offsets[v];
    let row_end   = row_offsets[v + 1u];
    let next_level = uniforms.current_level + 1i;

    for (var i = row_start; i < row_end; i = i + 1u) {
        let nb = col_indices[i];
        // CAS: only update if the neighbor is unvisited (distance == -1)
        let prev = atomicCompareExchangeWeak(&distances[nb], -1i, next_level);
        if (prev.old_value == -1i && prev.exchanged) {
            // Claim a slot in frontier_out
            let slot = atomicAdd(&frontier_out_count, 1u);
            atomicStore(&frontier_out[slot], nb);
        }
    }
}
"#;

/// Delta-stepping light-edge proposal shader.
///
/// Bindings (group 0):
/// - 0: `light_src`  — light edge source array (read-only, `array<u32>`)
/// - 1: `light_dst`  — light edge destination array (read-only, `array<u32>`)
/// - 2: `light_wt`   — light edge weight array (read-only, `array<f32>`)
/// - 3: `distances`  — current distance array in f32-bits (read-only, `array<u32>`)
/// - 4: `proposed`   — proposed update buffer in f32-bits (read-write, `array<atomic<u32>>`)
/// - 5: `uniforms`   — `{ n_light_edges: u32, _pad[3] }` (uniform)
///
/// Each invocation handles one light edge `(src, dst, weight)` where weight ≤ delta.
/// Proposes a shorter distance for `dst` using `atomicMin` on f32-bits.
pub const DELTA_LIGHT_WGSL: &str = r#"
struct DeltaLightUniforms {
    n_light_edges: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       light_src  : array<u32>;
@group(0) @binding(1) var<storage, read>       light_dst  : array<u32>;
@group(0) @binding(2) var<storage, read>       light_wt   : array<f32>;
@group(0) @binding(3) var<storage, read>       distances  : array<u32>;
@group(0) @binding(4) var<storage, read_write> proposed   : array<atomic<u32>>;
@group(0) @binding(5) var<uniform>             uniforms   : DeltaLightUniforms;

@compute @workgroup_size(256)
fn delta_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= uniforms.n_light_edges) {
        return;
    }
    let src = light_src[eid];
    let dst = light_dst[eid];
    let wt  = light_wt[eid];

    // Load src distance as f32 via bitcast (read-only snapshot)
    let d_src_bits = distances[src];
    let d_src = bitcast<f32>(d_src_bits);

    // Only relax if src is reachable (not INFINITY)
    if (d_src < 3.4028235e+38) {
        let nd = d_src + wt;
        let nd_bits = bitcast<u32>(nd);
        // atomicMin works correctly for non-negative IEEE 754 floats (bit ordering preserved)
        atomicMin(&proposed[dst], nd_bits);
    }
}
"#;

/// Delta-stepping apply-proposals shader.
///
/// Bindings (group 0):
/// - 0: `proposed`    — proposed distance buffer in f32-bits (read-only, `array<u32>`)
/// - 1: `distances`   — current distance array in f32-bits (read-write, `array<atomic<u32>>`)
/// - 2: `changed_flag`— convergence flag: set to 1 if any update applied (read-write, `atomic<u32>`)
/// - 3: `uniforms`    — `{ n_vertices: u32, _pad[3] }` (uniform)
///
/// Each invocation handles one vertex. Applies `proposed[v]` to `distances[v]`
/// using `atomicMin`. If the minimum changed, sets `changed_flag` to 1.
pub const DELTA_APPLY_WGSL: &str = r#"
struct DeltaApplyUniforms {
    n_vertices: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       proposed     : array<u32>;
@group(0) @binding(1) var<storage, read_write> distances    : array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> changed_flag : atomic<u32>;
@group(0) @binding(3) var<uniform>             uniforms     : DeltaApplyUniforms;

@compute @workgroup_size(256)
fn delta_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = gid.x;
    if (vid >= uniforms.n_vertices) {
        return;
    }
    let prop_bits = proposed[vid];
    // Only apply if proposal is not INFINITY (i.e., some light edge proposed an update)
    let prop_val = bitcast<f32>(prop_bits);
    if (prop_val < 3.4028235e+38) {
        // Attempt to update distances[vid] with atomicMin
        let old_bits = atomicMin(&distances[vid], prop_bits);
        // If the minimum actually changed (old > prop), mark convergence flag
        if (old_bits > prop_bits) {
            atomicStore(&changed_flag, 1u);
        }
    }
}
"#;

/// Heavy-edge relaxation shader for delta-stepping with convergence tracking.
///
/// Identical to Bellman-Ford but adds a `changed_flag` binding so the host
/// loop detects whether any heavy-edge relaxation updated a distance.
///
/// Bindings (group 0):
/// - 0: `edge_src`    — edge source array (read-only, `array<u32>`)
/// - 1: `edge_dst`    — edge destination array (read-only, `array<u32>`)
/// - 2: `edge_wt`     — edge weight array (read-only, `array<f32>`)
/// - 3: `distances`   — distance array in f32-bits (read-write, `array<atomic<u32>>`)
/// - 4: `changed_flag`— set to 1 if any update applied (read-write, `atomic<u32>`)
/// - 5: `uniforms`    — `{ n_edges: u32, _pad[3] }` (uniform)
pub const DELTA_HEAVY_WGSL: &str = r#"
struct HeavyUniforms {
    n_edges: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
};

@group(0) @binding(0) var<storage, read>       edge_src     : array<u32>;
@group(0) @binding(1) var<storage, read>       edge_dst     : array<u32>;
@group(0) @binding(2) var<storage, read>       edge_wt      : array<f32>;
@group(0) @binding(3) var<storage, read_write> distances    : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> changed_flag : atomic<u32>;
@group(0) @binding(5) var<uniform>             uniforms     : HeavyUniforms;

@compute @workgroup_size(256)
fn delta_heavy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= uniforms.n_edges) {
        return;
    }
    let src = edge_src[eid];
    let dst = edge_dst[eid];
    let wt  = edge_wt[eid];

    let d_src_bits = atomicLoad(&distances[src]);
    let d_src = bitcast<f32>(d_src_bits);

    if (d_src < 3.4028235e+38) {
        let nd = d_src + wt;
        let nd_bits = bitcast<u32>(nd);
        let old_bits = atomicMin(&distances[dst], nd_bits);
        // If the minimum actually changed, mark convergence flag
        if (old_bits > nd_bits) {
            atomicStore(&changed_flag, 1u);
        }
    }
}
"#;

/// Edge-parallel Bellman-Ford SSSP shader.
///
/// Bindings (group 0):
/// - 0: `edge_src`   — edge source array (read-only, `array<u32>`)
/// - 1: `edge_dst`   — edge destination array (read-only, `array<u32>`)
/// - 2: `edge_wt`    — edge weight array (read-only, `array<f32>`)
/// - 3: `distances`  — distance array in f32-bits (read-write, `array<atomic<u32>>`)
/// - 4: `uniforms`   — `{ n_edges: u32, source: u32 }` (uniform)
///
/// Each invocation handles one directed edge `(src, dst, weight)`.
/// Uses the f32-bits atomicMin trick: IEEE 754 ordering is preserved for
/// non-negative finite floats, so `atomicMin(&distances[dst], bits(d_src + w))`
/// correctly finds the minimum distance.
/// The host must initialize `distances[source] = 0.0f.to_bits()` and all
/// others to `f32::INFINITY.to_bits()` before dispatching.
pub const SSSP_BELLMAN_FORD_WGSL: &str = r#"
struct BfUniforms {
    n_edges: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
};

@group(0) @binding(0) var<storage, read>       edge_src  : array<u32>;
@group(0) @binding(1) var<storage, read>       edge_dst  : array<u32>;
@group(0) @binding(2) var<storage, read>       edge_wt   : array<f32>;
@group(0) @binding(3) var<storage, read_write> distances : array<atomic<u32>>;
@group(0) @binding(4) var<uniform>             uniforms  : BfUniforms;

@compute @workgroup_size(64)
fn sssp_bellman_ford(@builtin(global_invocation_id) gid: vec3<u32>) {
    let eid = gid.x;
    if (eid >= uniforms.n_edges) {
        return;
    }
    let src = edge_src[eid];
    let dst = edge_dst[eid];
    let wt  = edge_wt[eid];

    // Load src distance as f32 via bitcast
    let d_src_bits = atomicLoad(&distances[src]);
    let d_src = bitcast<f32>(d_src_bits);

    // Only relax if src is reachable (not INFINITY)
    if (d_src < 3.4028235e+38) {
        let nd = d_src + wt;
        let nd_bits = bitcast<u32>(nd);
        // atomicMin works correctly for non-negative IEEE 754 floats
        atomicMin(&distances[dst], nd_bits);
    }
}
"#;
