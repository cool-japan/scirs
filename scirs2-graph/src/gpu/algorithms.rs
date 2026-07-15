//! Parallel BFS and SSSP algorithms with GPU dispatch.
//!
//! All algorithms accept graphs in standard formats (CSR or adjacency lists)
//! and dispatch to the GPU (wgpu) when the `wgpu` feature is enabled and a
//! compatible adapter is present.  When no adapter is available, or when the
//! edge count is below the threshold (4096), the call falls back to the
//! CPU-parallel implementations in `super::parallel`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::{GraphError, Result};

pub(super) use super::parallel::{parallel_bellman_ford_atomic, parallel_bfs_atomic};

/// GPU edge-count threshold below which CPU execution is preferred.
const GPU_EDGE_THRESHOLD: usize = 4096;

/// Backend selection for GPU-accelerated graph algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuGraphBackend {
    /// CPU-parallel execution using atomic operations.
    CpuParallel,
    /// GPU backend via wgpu (with CPU fallback when no adapter is found).
    Gpu,
}

/// Configuration for GPU/parallel BFS and SSSP algorithms.
#[derive(Debug, Clone)]
pub struct GpuBfsConfig {
    /// Backend to use. Default: [`GpuGraphBackend::CpuParallel`].
    pub backend: GpuGraphBackend,
    /// Frontier chunk size for parallel processing. Default: 1024.
    pub chunk_size: usize,
}

impl Default for GpuBfsConfig {
    fn default() -> Self {
        Self {
            backend: GpuGraphBackend::CpuParallel,
            chunk_size: 1024,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BFS
// ─────────────────────────────────────────────────────────────────────────────

/// BFS from `source` on a CSR graph. Returns distances (usize::MAX = unreachable).
///
/// When `config.backend == Gpu` and enough edges are present (≥4096), dispatches
/// to the wgpu shader. Falls back to CPU-parallel on missing adapter or compile
/// error. When `config.backend == CpuParallel`, always uses CPU-parallel.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `source >= n` or CSR arrays are inconsistent.
pub fn gpu_bfs(
    row_ptr: &[usize],
    col_idx: &[usize],
    source: usize,
    config: &GpuBfsConfig,
) -> Result<Vec<usize>> {
    if row_ptr.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "row_ptr".to_string(),
            value: format!("len={}", row_ptr.len()),
            expected: "at least 2 elements (n+1)".to_string(),
            context: "gpu_bfs".to_string(),
        });
    }

    let n = row_ptr.len() - 1;

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{source}"),
            expected: format!("0..{n}"),
            context: "gpu_bfs".to_string(),
        });
    }

    if let Some(&last) = row_ptr.last() {
        if last > col_idx.len() {
            return Err(GraphError::InvalidParameter {
                param: "row_ptr/col_idx".to_string(),
                value: format!("row_ptr last={}, col_idx len={}", last, col_idx.len()),
                expected: "row_ptr[n] <= col_idx.len()".to_string(),
                context: "gpu_bfs".to_string(),
            });
        }
    }

    match config.backend {
        GpuGraphBackend::CpuParallel => Ok(parallel_bfs_atomic(row_ptr, col_idx, source)),
        GpuGraphBackend::Gpu => {
            #[cfg(feature = "wgpu")]
            {
                let n_edges = col_idx.len();
                if n_edges >= GPU_EDGE_THRESHOLD {
                    match wgpu_bfs(row_ptr, col_idx, source) {
                        Ok(dist) => return Ok(dist),
                        Err(_) => {
                            // No adapter or dispatch error — fall through to CPU
                        }
                    }
                }
            }
            // CPU fallback (below threshold or no adapter)
            Ok(parallel_bfs_atomic(row_ptr, col_idx, source))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bellman-Ford SSSP
// ─────────────────────────────────────────────────────────────────────────────

/// Bellman-Ford SSSP on a CSR graph with edge weights.
///
/// Detects negative-weight cycles in the CPU path: returns an error if any
/// cycle with negative total weight is reachable from `source`.
///
/// When `config.backend == Gpu` and enough edges are present (≥4096) and all
/// weights are non-negative, dispatches to the wgpu edge-parallel shader.
/// Falls back to CPU sequential on negative weights, missing adapter, or below threshold.
///
/// # Errors
/// Returns [`GraphError::AlgorithmFailure`] on negative cycle detection (CPU path),
/// [`GraphError::InvalidParameter`] for bad inputs.
pub fn gpu_sssp_bellman_ford(
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f64],
    source: usize,
    config: &GpuBfsConfig,
) -> Result<Vec<f64>> {
    if row_ptr.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "row_ptr".to_string(),
            value: format!("len={}", row_ptr.len()),
            expected: "at least 2 elements (n+1)".to_string(),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    let n = row_ptr.len() - 1;

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{source}"),
            expected: format!("0..{n}"),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    if col_idx.len() != weights.len() {
        return Err(GraphError::InvalidParameter {
            param: "weights".to_string(),
            value: format!("len={}", weights.len()),
            expected: format!("same length as col_idx ({})", col_idx.len()),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    let has_negative = weights.iter().any(|&w| w < 0.0);

    match config.backend {
        GpuGraphBackend::CpuParallel => {
            if has_negative {
                // GPU-parallel trick only works for non-negative; use sequential with cycle detection
                bellman_ford_sequential(row_ptr, col_idx, weights, source, n)
            } else {
                // Convert f64 → f32 for parallel path, then back to f64
                let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
                match parallel_bellman_ford_atomic(row_ptr, col_idx, &weights_f32, source) {
                    Ok(result) => Ok(result.into_iter().map(|v| v as f64).collect()),
                    Err(_) => bellman_ford_sequential(row_ptr, col_idx, weights, source, n),
                }
            }
        }
        GpuGraphBackend::Gpu => {
            #[cfg(feature = "wgpu")]
            {
                let n_edges = col_idx.len();
                if !has_negative && n_edges >= GPU_EDGE_THRESHOLD {
                    let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
                    match wgpu_bellman_ford(row_ptr, col_idx, &weights_f32, source, n) {
                        Ok(dist_f32) => {
                            return Ok(dist_f32.into_iter().map(|v| v as f64).collect());
                        }
                        Err(_) => {
                            // No adapter or dispatch error — fall through to CPU
                        }
                    }
                }
            }
            // CPU sequential fallback (supports negative weights + cycle detection)
            bellman_ford_sequential(row_ptr, col_idx, weights, source, n)
        }
    }
}

/// Sequential Bellman-Ford with negative-cycle detection (CPU-only fallback).
fn bellman_ford_sequential(
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f64],
    source: usize,
    n: usize,
) -> Result<Vec<f64>> {
    let has_negative = weights.iter().any(|&w| w < 0.0);

    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;

    for _ in 0..(n.saturating_sub(1)) {
        let mut changed = false;
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_idx[idx];
                let w = weights[idx];
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Detect negative cycles
    if has_negative {
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_idx[idx];
                let w = weights[idx];
                if dist[u] + w < dist[v] {
                    return Err(GraphError::AlgorithmFailure {
                        algorithm: "gpu_sssp_bellman_ford".to_string(),
                        reason: "negative weight cycle detected".to_string(),
                        iterations: n,
                        tolerance: 0.0,
                    });
                }
            }
        }
    }

    Ok(dist)
}

// ─────────────────────────────────────────────────────────────────────────────
// Delta-stepping SSSP
// ─────────────────────────────────────────────────────────────────────────────

/// Delta-stepping SSSP on an adjacency-list graph.
///
/// Partitions edges into light (weight ≤ delta) and heavy categories.
/// When `config.backend == Gpu` and enough edges are present (≥4096), dispatches
/// true GPU delta-stepping using two WGSL kernels:
/// - `delta_light`: proposes updates for light edges via `atomicMin` on f32-bits
/// - `delta_apply`: applies proposals to the distance buffer
///
/// Heavy edges are relaxed once per iteration using the Bellman-Ford kernel.
/// Convergence is detected via a GPU-side flag buffer.
/// Falls back to sequential Dijkstra for small graphs or when no GPU adapter is found.
///
/// The adaptive delta heuristic uses `max_weight / sqrt(n)` when `delta` is the
/// special sentinel value 0.0 (see `gpu_sssp_delta_stepping_adaptive`).
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `delta <= 0`, `source` is out
/// of range, or negative edge weights are detected.
pub fn gpu_sssp_delta_stepping(
    adj: &[Vec<(usize, f64)>],
    source: usize,
    delta: f64,
    config: &GpuBfsConfig,
) -> Result<Vec<f64>> {
    let n = adj.len();

    if n == 0 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: "len=0".to_string(),
            expected: "non-empty graph".to_string(),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{source}"),
            expected: format!("0..{n}"),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    if delta <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "delta".to_string(),
            value: format!("{delta}"),
            expected: "positive value".to_string(),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    // Reject negative weights
    for (u, nbrs) in adj.iter().enumerate() {
        for &(_, w) in nbrs {
            if w < 0.0 {
                return Err(GraphError::InvalidParameter {
                    param: "weights".to_string(),
                    value: format!("negative weight on edge from {u}"),
                    expected: "non-negative edge weights for delta-stepping".to_string(),
                    context: "gpu_sssp_delta_stepping".to_string(),
                });
            }
        }
    }

    // Build CSR from adjacency list
    let mut row_ptr = vec![0usize; n + 1];
    for (u, nbrs) in adj.iter().enumerate() {
        row_ptr[u + 1] = row_ptr[u] + nbrs.len();
    }
    let total_edges = row_ptr[n];
    let mut col_idx = Vec::with_capacity(total_edges);
    let mut wts = Vec::with_capacity(total_edges);
    for nbrs in adj {
        for &(v, w) in nbrs {
            col_idx.push(v);
            wts.push(w);
        }
    }

    match config.backend {
        GpuGraphBackend::CpuParallel => {
            // Sequential Dijkstra (equivalent, correct for non-negative weights)
            Ok(dijkstra_sequential(adj, source))
        }
        GpuGraphBackend::Gpu => {
            // True GPU delta-stepping: light-edge proposal + apply + heavy-edge BF pass.
            #[cfg(feature = "wgpu")]
            {
                if total_edges >= GPU_EDGE_THRESHOLD {
                    match wgpu_delta_stepping(adj, source, delta) {
                        Ok(dist_f32) => {
                            return Ok(dist_f32.into_iter().map(|v| v as f64).collect());
                        }
                        Err(_) => {
                            // No adapter or shader error — fall through to CPU
                        }
                    }
                }
            }
            Ok(dijkstra_sequential(adj, source))
        }
    }
}

/// Sequential Dijkstra reference implementation.
fn dijkstra_sequential(adj: &[Vec<(usize, f64)>], src: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    dist[src] = 0.0;
    let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
    heap.push((Reverse(0), src));
    while let Some((Reverse(d_scaled), u)) = heap.pop() {
        let d = d_scaled as f64 * 1e-9;
        if d > dist[u] + 1e-12 {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = dist[u] + w;
            if nd < dist[v] - 1e-12 {
                dist[v] = nd;
                heap.push((Reverse((nd * 1e9) as u64), v));
            }
        }
    }
    dist
}

// ─────────────────────────────────────────────────────────────────────────────
// wgpu dispatch (compiled only when `wgpu` feature + scirs2-core/wgpu available)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "wgpu")]
fn wgpu_bfs(
    row_ptr: &[usize],
    col_idx: &[usize],
    source: usize,
) -> std::result::Result<Vec<usize>, String> {
    use wgpu::{
        util::DeviceExt as _, Backends, BindGroupDescriptor, BindGroupEntry,
        BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
        BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
        DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits, PowerPreference,
        RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    };

    let n = row_ptr.len() - 1;

    // Validate sizes fit in u32
    if n > u32::MAX as usize {
        return Err("graph too large for u32 indexing".into());
    }
    if col_idx.len() > u32::MAX as usize {
        return Err("edge list too large for u32 indexing".into());
    }

    // Acquire adapter and device
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| "no wgpu adapter available".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-graph-bfs"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| format!("device creation failed: {e}"))?;

    // Convert CSR to u32
    let row_offsets_u32: Vec<u32> = row_ptr.iter().map(|&x| x as u32).collect();
    let col_indices_u32: Vec<u32> = col_idx.iter().map(|&x| x as u32).collect();

    // Distances: init all to -1 (unvisited)
    let mut distances_i32: Vec<i32> = vec![-1i32; n];
    distances_i32[source] = 0;

    // Max frontier size = n vertices
    let frontier_capacity = n;

    // Create device buffers
    let buf_row = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bfs-row-offsets"),
        contents: bytemuck::cast_slice(&row_offsets_u32),
        usage: BufferUsages::STORAGE,
    });
    let buf_col = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bfs-col-indices"),
        contents: bytemuck::cast_slice(&col_indices_u32),
        usage: BufferUsages::STORAGE,
    });
    let buf_dist = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bfs-distances"),
        contents: bytemuck::cast_slice(&distances_i32),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    // Frontier in/out buffers (u32, one slot per vertex max)
    let frontier_bytes = (frontier_capacity * std::mem::size_of::<u32>()) as u64;
    let buf_frontier_in = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-frontier-in"),
        size: frontier_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_frontier_out = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-frontier-out"),
        size: frontier_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Frontier count (atomic<u32>) — 4 bytes
    let buf_frontier_count = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-frontier-count"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniform buffer: { frontier_size: u32, current_level: i32 }
    // Pad to 16 bytes for uniform alignment
    let buf_uniforms = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-uniforms"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Staging buffers for readback
    let dist_byte_len = (n * std::mem::size_of::<i32>()) as u64;
    let buf_dist_staging = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-dist-staging"),
        size: dist_byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_count_staging = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-count-staging"),
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_frontier_out_staging = device.create_buffer(&BufferDescriptor {
        label: Some("bfs-frontier-out-staging"),
        size: frontier_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Compile shader
    use super::wgpu_shaders::BFS_FRONTIER_WGSL;
    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bfs-frontier-shader"),
        source: ShaderSource::Wgsl(BFS_FRONTIER_WGSL.into()),
    });

    // Build bind group layout matching shader bindings 0-6
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bfs-bgl"),
        entries: &[
            bgl_storage_ro(0), // row_offsets
            bgl_storage_ro(1), // col_indices
            bgl_storage_rw(2), // distances (atomic)
            bgl_storage_ro(3), // frontier_in
            bgl_storage_rw(4), // frontier_out (atomic)
            bgl_storage_rw(5), // frontier_out_count (atomic)
            bgl_uniform(6),    // uniforms
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bfs-layout"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bfs-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("bfs_frontier"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Seed: frontier_in = [source], frontier_count = 0
    let source_u32 = source as u32;
    let source_bytes = source_u32.to_le_bytes();
    queue.write_buffer(&buf_frontier_in, 0, &source_bytes);
    queue.write_buffer(&buf_frontier_count, 0, &0u32.to_le_bytes());

    let mut frontier_size: u32 = 1;
    let mut current_level: i32 = 0;

    // BFS level loop — one dispatch per frontier level
    while frontier_size > 0 {
        // Write uniforms
        let uniforms_data: [u8; 16] = {
            let mut buf = [0u8; 16];
            buf[0..4].copy_from_slice(&frontier_size.to_le_bytes());
            buf[4..8].copy_from_slice(&current_level.to_le_bytes());
            buf
        };
        queue.write_buffer(&buf_uniforms, 0, &uniforms_data);
        queue.write_buffer(&buf_frontier_count, 0, &0u32.to_le_bytes());

        // Clear frontier_out (not strictly required but defensive)
        let zero_frontier: Vec<u8> = vec![0u8; frontier_capacity * 4];
        queue.write_buffer(&buf_frontier_out, 0, &zero_frontier);

        // Bind group (static for this iteration)
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("bfs-bg"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf_row.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buf_col.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: buf_dist.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: buf_frontier_in.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: buf_frontier_out.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: buf_frontier_count.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: buf_uniforms.as_entire_binding(),
                },
            ],
        });

        // Dispatch one workgroup per 64 frontier vertices
        let workgroups = frontier_size.div_ceil(64);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        // Copy count → staging
        encoder.copy_buffer_to_buffer(&buf_frontier_count, 0, &buf_count_staging, 0, 4);
        // Copy frontier_out → staging (we'll read how many needed from count)
        encoder.copy_buffer_to_buffer(
            &buf_frontier_out,
            0,
            &buf_frontier_out_staging,
            0,
            frontier_bytes,
        );
        queue.submit([encoder.finish()]);

        // Wait for GPU
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll error: {e:?}"))?;

        // Read back count
        let next_count = read_u32_from_staging(&device, &buf_count_staging)?;
        if next_count == 0 {
            break;
        }

        // Read back frontier_out (only as many as next_count)
        let next_frontier =
            read_u32_vec_from_staging(&device, &buf_frontier_out_staging, next_count as usize)?;

        // Copy frontier_out → frontier_in for next iteration
        let next_bytes: Vec<u8> = next_frontier
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        queue.write_buffer(&buf_frontier_in, 0, &next_bytes);

        frontier_size = next_count;
        current_level += 1;
    }

    // Read back distances
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buf_dist, 0, &buf_dist_staging, 0, dist_byte_len);
    queue.submit([encoder.finish()]);

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll error during dist readback: {e:?}"))?;

    let dist_i32 = read_i32_vec_from_staging(&device, &buf_dist_staging, n)?;

    // Convert: -1 → usize::MAX, otherwise cast to usize
    let dist_usize: Vec<usize> = dist_i32
        .into_iter()
        .map(|d| if d < 0 { usize::MAX } else { d as usize })
        .collect();

    Ok(dist_usize)
}

#[cfg(feature = "wgpu")]
fn wgpu_bellman_ford(
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f32],
    source: usize,
    n: usize,
) -> std::result::Result<Vec<f32>, String> {
    use wgpu::{
        util::DeviceExt as _, Backends, BindGroupDescriptor, BindGroupEntry,
        BindGroupLayoutDescriptor, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
        ComputePassDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    };

    if n > u32::MAX as usize || col_idx.len() > u32::MAX as usize {
        return Err("graph too large for u32 indexing".into());
    }

    // Build flat edge list from CSR
    let n_edges = col_idx.len();
    let mut edge_src: Vec<u32> = Vec::with_capacity(n_edges);
    let mut edge_dst: Vec<u32> = Vec::with_capacity(n_edges);
    let mut edge_wt: Vec<f32> = Vec::with_capacity(n_edges);
    for u in 0..n {
        let start = row_ptr[u];
        let end = row_ptr[u + 1];
        for i in start..end {
            edge_src.push(u as u32);
            edge_dst.push(col_idx[i] as u32);
            edge_wt.push(weights[i]);
        }
    }

    // Acquire adapter and device
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| "no wgpu adapter available".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-graph-bf"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| format!("device creation failed: {e}"))?;

    // Init distances: source=0.0, others=INFINITY (as u32 bits)
    let inf_bits = f32::INFINITY.to_bits();
    let mut dist_bits: Vec<u32> = vec![inf_bits; n];
    dist_bits[source] = 0u32;

    // Create buffers
    let buf_src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bf-edge-src"),
        contents: bytemuck::cast_slice(&edge_src),
        usage: BufferUsages::STORAGE,
    });
    let buf_dst = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bf-edge-dst"),
        contents: bytemuck::cast_slice(&edge_dst),
        usage: BufferUsages::STORAGE,
    });
    let buf_wt = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bf-edge-wt"),
        contents: bytemuck::cast_slice(&edge_wt),
        usage: BufferUsages::STORAGE,
    });
    let buf_dist = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bf-distances"),
        contents: bytemuck::cast_slice(&dist_bits),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    // Uniform: { n_edges: u32, _pad[3] }
    let uniforms: [u32; 4] = [n_edges as u32, 0, 0, 0];
    let buf_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bf-uniforms"),
        contents: bytemuck::cast_slice(&uniforms),
        usage: BufferUsages::UNIFORM,
    });

    // Compile shader
    use super::wgpu_shaders::SSSP_BELLMAN_FORD_WGSL;
    let shader_module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("bf-shader"),
        source: ShaderSource::Wgsl(SSSP_BELLMAN_FORD_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("bf-bgl"),
        entries: &[
            bgl_storage_ro(0), // edge_src
            bgl_storage_ro(1), // edge_dst
            bgl_storage_ro(2), // edge_wt
            bgl_storage_rw(3), // distances (atomic)
            bgl_uniform(4),    // uniforms
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bf-layout"),
        bind_group_layouts: &[Some(&bgl)],
        ..Default::default()
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bf-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("sssp_bellman_ford"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("bf-bg"),
        layout: &bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_src.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_dst.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_wt.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buf_dist.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: buf_uniforms.as_entire_binding(),
            },
        ],
    });

    let workgroups = (n_edges as u32).div_ceil(64);

    // n-1 Bellman-Ford passes
    for _ in 0..(n.saturating_sub(1)) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit([encoder.finish()]);

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll error: {e:?}"))?;
    }

    // Readback
    let dist_byte_len = (n * std::mem::size_of::<u32>()) as u64;
    let buf_staging = device.create_buffer(&BufferDescriptor {
        label: Some("bf-staging"),
        size: dist_byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buf_dist, 0, &buf_staging, 0, dist_byte_len);
    queue.submit([encoder.finish()]);

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll error during readback: {e:?}"))?;

    let result_bits = read_u32_vec_from_staging(&device, &buf_staging, n)?;
    Ok(result_bits.into_iter().map(f32::from_bits).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// True GPU delta-stepping dispatch (light/heavy partition + atomicMin)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "wgpu")]
fn wgpu_delta_stepping(
    adj: &[Vec<(usize, f64)>],
    source: usize,
    delta: f64,
) -> std::result::Result<Vec<f32>, String> {
    use super::wgpu_shaders::{DELTA_APPLY_WGSL, DELTA_HEAVY_WGSL, DELTA_LIGHT_WGSL};
    use wgpu::{
        util::DeviceExt as _, Backends, BindGroupDescriptor, BindGroupEntry,
        BindGroupLayoutDescriptor, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
        ComputePassDescriptor, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
        PowerPreference, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    };

    let n = adj.len();
    if n > u32::MAX as usize {
        return Err("graph too large for u32 indexing".into());
    }

    // ── CPU-side preprocessing: partition edges into light and heavy ──────────
    let delta_f32 = delta as f32;
    let mut light_src: Vec<u32> = Vec::new();
    let mut light_dst: Vec<u32> = Vec::new();
    let mut light_wt: Vec<f32> = Vec::new();
    let mut heavy_src: Vec<u32> = Vec::new();
    let mut heavy_dst: Vec<u32> = Vec::new();
    let mut heavy_wt: Vec<f32> = Vec::new();
    for (u, nbrs) in adj.iter().enumerate() {
        for &(v, w) in nbrs {
            let w32 = w as f32;
            if w32 <= delta_f32 {
                light_src.push(u as u32);
                light_dst.push(v as u32);
                light_wt.push(w32);
            } else {
                heavy_src.push(u as u32);
                heavy_dst.push(v as u32);
                heavy_wt.push(w32);
            }
        }
    }

    // ── Acquire adapter and device ────────────────────────────────────────────
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|_| "no wgpu adapter available".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("scirs2-graph-delta"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| format!("device creation failed: {e}"))?;

    // ── Init distance buffer ───────────────────────────────────────────────────
    let inf_bits = f32::INFINITY.to_bits();
    let mut dist_bits: Vec<u32> = vec![inf_bits; n];
    dist_bits[source] = 0u32; // distance[source] = 0.0f.bits()

    // ── Create shared distance buffer ─────────────────────────────────────────
    let buf_dist = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-distances"),
        contents: bytemuck::cast_slice(&dist_bits),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });

    // ── Create proposed buffer (inf initially, reset each light phase) ─────────
    let proposed_init: Vec<u32> = vec![inf_bits; n];
    let buf_proposed = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-proposed"),
        contents: bytemuck::cast_slice(&proposed_init),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    // ── Convergence flag buffer ────────────────────────────────────────────────
    let buf_changed = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-changed"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    });
    let buf_changed_staging = device.create_buffer(&BufferDescriptor {
        label: Some("ds-changed-staging"),
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Light edge buffers ────────────────────────────────────────────────────
    let n_light = light_src.len();
    // Handle empty light edge list (create minimal placeholder buffers)
    let (buf_light_src, buf_light_dst, buf_light_wt) = if n_light > 0 {
        let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-src"),
            contents: bytemuck::cast_slice(&light_src),
            usage: BufferUsages::STORAGE,
        });
        let d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-dst"),
            contents: bytemuck::cast_slice(&light_dst),
            usage: BufferUsages::STORAGE,
        });
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-wt"),
            contents: bytemuck::cast_slice(&light_wt),
            usage: BufferUsages::STORAGE,
        });
        (s, d, w)
    } else {
        // Placeholder 4-byte buffers so bind group creation doesn't fail
        let placeholder: [u32; 1] = [0];
        let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-src-ph"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: BufferUsages::STORAGE,
        });
        let d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-dst-ph"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: BufferUsages::STORAGE,
        });
        let w_ph: [f32; 1] = [0.0];
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-light-wt-ph"),
            contents: bytemuck::cast_slice(&w_ph),
            usage: BufferUsages::STORAGE,
        });
        (s, d, w)
    };

    // ── Heavy edge buffers ────────────────────────────────────────────────────
    let n_heavy = heavy_src.len();
    let (buf_heavy_src, buf_heavy_dst, buf_heavy_wt) = if n_heavy > 0 {
        let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-src"),
            contents: bytemuck::cast_slice(&heavy_src),
            usage: BufferUsages::STORAGE,
        });
        let d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-dst"),
            contents: bytemuck::cast_slice(&heavy_dst),
            usage: BufferUsages::STORAGE,
        });
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-wt"),
            contents: bytemuck::cast_slice(&heavy_wt),
            usage: BufferUsages::STORAGE,
        });
        (s, d, w)
    } else {
        let placeholder: [u32; 1] = [0];
        let s = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-src-ph"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: BufferUsages::STORAGE,
        });
        let d = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-dst-ph"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: BufferUsages::STORAGE,
        });
        let w_ph: [f32; 1] = [0.0];
        let w = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ds-heavy-wt-ph"),
            contents: bytemuck::cast_slice(&w_ph),
            usage: BufferUsages::STORAGE,
        });
        (s, d, w)
    };

    // ── Compile shaders ───────────────────────────────────────────────────────
    let light_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ds-light-shader"),
        source: ShaderSource::Wgsl(DELTA_LIGHT_WGSL.into()),
    });
    let apply_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ds-apply-shader"),
        source: ShaderSource::Wgsl(DELTA_APPLY_WGSL.into()),
    });
    let heavy_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ds-heavy-shader"),
        source: ShaderSource::Wgsl(DELTA_HEAVY_WGSL.into()),
    });

    // ── Build bind group layouts ──────────────────────────────────────────────
    // Light kernel: bindings 0-5 (src, dst, wt, distances_ro, proposed_rw, uniforms)
    let light_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ds-light-bgl"),
        entries: &[
            bgl_storage_ro(0), // light_src
            bgl_storage_ro(1), // light_dst
            bgl_storage_ro(2), // light_wt
            bgl_storage_ro(3), // distances (read-only snapshot)
            bgl_storage_rw(4), // proposed (atomic write)
            bgl_uniform(5),    // uniforms
        ],
    });

    // Apply kernel: bindings 0-3 (proposed_ro, distances_rw, changed_rw, uniforms)
    let apply_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ds-apply-bgl"),
        entries: &[
            bgl_storage_ro(0), // proposed
            bgl_storage_rw(1), // distances (atomic update)
            bgl_storage_rw(2), // changed_flag (atomic)
            bgl_uniform(3),    // uniforms
        ],
    });

    // Heavy kernel (DELTA_HEAVY): 0-5 (src, dst, wt, distances_rw, changed_flag_rw, uniforms)
    let heavy_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ds-heavy-bgl"),
        entries: &[
            bgl_storage_ro(0), // heavy_src
            bgl_storage_ro(1), // heavy_dst
            bgl_storage_ro(2), // heavy_wt
            bgl_storage_rw(3), // distances (atomic)
            bgl_storage_rw(4), // changed_flag (atomic) — tracks heavy-edge updates
            bgl_uniform(5),    // uniforms
        ],
    });

    // ── Build pipelines ───────────────────────────────────────────────────────
    let light_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ds-light-layout"),
        bind_group_layouts: &[Some(&light_bgl)],
        ..Default::default()
    });
    let apply_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ds-apply-layout"),
        bind_group_layouts: &[Some(&apply_bgl)],
        ..Default::default()
    });
    let heavy_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ds-heavy-layout"),
        bind_group_layouts: &[Some(&heavy_bgl)],
        ..Default::default()
    });

    let light_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ds-light-pipeline"),
        layout: Some(&light_pl_layout),
        module: &light_shader,
        entry_point: Some("delta_light"),
        compilation_options: Default::default(),
        cache: None,
    });
    let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ds-apply-pipeline"),
        layout: Some(&apply_pl_layout),
        module: &apply_shader,
        entry_point: Some("delta_apply"),
        compilation_options: Default::default(),
        cache: None,
    });
    let heavy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("ds-heavy-pipeline"),
        layout: Some(&heavy_pl_layout),
        module: &heavy_shader,
        entry_point: Some("delta_heavy"),
        compilation_options: Default::default(),
        cache: None,
    });

    // ── Uniform buffers ───────────────────────────────────────────────────────
    let light_uniforms: [u32; 4] = [n_light as u32, 0, 0, 0];
    let buf_light_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-light-uniforms"),
        contents: bytemuck::cast_slice(&light_uniforms),
        usage: BufferUsages::UNIFORM,
    });

    let apply_uniforms: [u32; 4] = [n as u32, 0, 0, 0];
    let buf_apply_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-apply-uniforms"),
        contents: bytemuck::cast_slice(&apply_uniforms),
        usage: BufferUsages::UNIFORM,
    });

    let heavy_uniforms: [u32; 4] = [n_heavy as u32, 0, 0, 0];
    let buf_heavy_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ds-heavy-uniforms"),
        contents: bytemuck::cast_slice(&heavy_uniforms),
        usage: BufferUsages::UNIFORM,
    });

    // ── Build bind groups ─────────────────────────────────────────────────────
    let light_bg = device.create_bind_group(&BindGroupDescriptor {
        label: Some("ds-light-bg"),
        layout: &light_bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_light_src.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_light_dst.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_light_wt.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buf_dist.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: buf_proposed.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: buf_light_uniforms.as_entire_binding(),
            },
        ],
    });

    let apply_bg = device.create_bind_group(&BindGroupDescriptor {
        label: Some("ds-apply-bg"),
        layout: &apply_bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_proposed.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_dist.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_changed.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buf_apply_uniforms.as_entire_binding(),
            },
        ],
    });

    let heavy_bg = device.create_bind_group(&BindGroupDescriptor {
        label: Some("ds-heavy-bg"),
        layout: &heavy_bgl,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_heavy_src.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_heavy_dst.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_heavy_wt.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: buf_dist.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: buf_changed.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: buf_heavy_uniforms.as_entire_binding(),
            },
        ],
    });

    // ── Workgroup counts ──────────────────────────────────────────────────────
    let light_wg = (n_light as u32).saturating_add(255) / 256;
    let apply_wg = (n as u32).saturating_add(255) / 256;
    let heavy_wg = (n_heavy as u32).saturating_add(255) / 256;

    // ── Main convergence loop ─────────────────────────────────────────────────
    // Maximum iterations bounded by n to prevent infinite loops in edge cases.
    let max_iters = n + 1;
    for _ in 0..max_iters {
        // Reset proposed buffer to INFINITY before each light phase
        let inf_reset: Vec<u32> = vec![inf_bits; n];
        queue.write_buffer(&buf_proposed, 0, bytemuck::cast_slice(&inf_reset));
        // Reset changed flag to 0
        queue.write_buffer(&buf_changed, 0, &0u32.to_le_bytes());

        // Phase 1: Light edge proposal
        if n_light > 0 {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&light_pipeline);
                pass.set_bind_group(0, &light_bg, &[]);
                pass.dispatch_workgroups(light_wg.max(1), 1, 1);
            }
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU poll (light): {e:?}"))?;
        }

        // Phase 2: Apply proposals to distances, track convergence
        {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&apply_pipeline);
                pass.set_bind_group(0, &apply_bg, &[]);
                pass.dispatch_workgroups(apply_wg.max(1), 1, 1);
            }
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU poll (apply): {e:?}"))?;
        }

        // Phase 3: Heavy edge relaxation (one pass, like Bellman-Ford step)
        if n_heavy > 0 {
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&heavy_pipeline);
                pass.set_bind_group(0, &heavy_bg, &[]);
                pass.dispatch_workgroups(heavy_wg.max(1), 1, 1);
            }
            // Copy changed flag to staging for readback
            encoder.copy_buffer_to_buffer(&buf_changed, 0, &buf_changed_staging, 0, 4);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU poll (heavy): {e:?}"))?;
        } else {
            // No heavy edges: copy changed flag to staging directly
            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&buf_changed, 0, &buf_changed_staging, 0, 4);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| format!("GPU poll (no-heavy): {e:?}"))?;
        }

        // Read convergence flag
        let changed = read_u32_from_staging(&device, &buf_changed_staging)?;
        if changed == 0 {
            break;
        }
    }

    // ── Read back final distances ─────────────────────────────────────────────
    let dist_byte_len = (n * std::mem::size_of::<u32>()) as u64;
    let buf_staging = device.create_buffer(&BufferDescriptor {
        label: Some("ds-dist-staging"),
        size: dist_byte_len,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(&buf_dist, 0, &buf_staging, 0, dist_byte_len);
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("GPU poll (readback): {e:?}"))?;

    let result_bits = read_u32_vec_from_staging(&device, &buf_staging, n)?;
    Ok(result_bits.into_iter().map(f32::from_bits).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions for bind group layout entries
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "wgpu")]
fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "wgpu")]
fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(feature = "wgpu")]
fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU buffer readback helpers
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "wgpu")]
fn read_u32_from_staging(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
) -> std::result::Result<u32, String> {
    let slice = staging.slice(0..4);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll error: {e:?}"))?;
    rx.recv()
        .map_err(|_| "channel closed".to_string())?
        .map_err(|e| format!("map_async failed: {e:?}"))?;
    let mapped = slice.get_mapped_range();
    let val = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
    drop(mapped);
    staging.unmap();
    Ok(val)
}

#[cfg(feature = "wgpu")]
fn read_u32_vec_from_staging(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> std::result::Result<Vec<u32>, String> {
    let byte_len = (count * 4) as u64;
    let slice = staging.slice(0..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll error: {e:?}"))?;
    rx.recv()
        .map_err(|_| "channel closed".to_string())?
        .map_err(|e| format!("map_async failed: {e:?}"))?;
    let mapped = slice.get_mapped_range();
    let result = mapped
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    Ok(result)
}

#[cfg(feature = "wgpu")]
fn read_i32_vec_from_staging(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    count: usize,
) -> std::result::Result<Vec<i32>, String> {
    let byte_len = (count * 4) as u64;
    let slice = staging.slice(0..byte_len);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll error: {e:?}"))?;
    rx.recv()
        .map_err(|_| "channel closed".to_string())?
        .map_err(|e| format!("map_async failed: {e:?}"))?;
    let mapped = slice.get_mapped_range();
    let result = mapped
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    drop(mapped);
    staging.unmap();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_csr(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + adj[i].len();
        }
        let col_idx: Vec<usize> = adj.into_iter().flatten().collect();
        (row_ptr, col_idx)
    }

    fn build_csr_directed(
        n: usize,
        edges: &[(usize, usize, f64)],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(u, v, w) in edges {
            adj[u].push((v, w));
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + adj[i].len();
        }
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();
        for nbrs in adj {
            for (v, w) in nbrs {
                col_idx.push(v);
                weights.push(w);
            }
        }
        (row_ptr, col_idx, weights)
    }

    fn dijkstra_ref(adj: &[Vec<(usize, f64)>], src: usize) -> Vec<f64> {
        let n = adj.len();
        let mut dist = vec![f64::INFINITY; n];
        dist[src] = 0.0;
        let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        heap.push((Reverse(0), src));
        while let Some((Reverse(d), u)) = heap.pop() {
            let d = d as f64 * 1e-9;
            if d > dist[u] + 1e-12 {
                continue;
            }
            for &(v, w) in &adj[u] {
                let nd = dist[u] + w;
                if nd < dist[v] - 1e-12 {
                    dist[v] = nd;
                    heap.push((Reverse((nd * 1e9) as u64), v));
                }
            }
        }
        dist
    }

    #[test]
    fn test_gpu_bfs_path_graph() {
        let (rp, ci) = build_csr(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_gpu_bfs_connected() {
        let edges: Vec<(usize, usize)> = (0..5usize)
            .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
            .collect();
        let (rp, ci) = build_csr(5, &edges);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        for i in 1..5 {
            assert_eq!(dist[i], 1);
        }
    }

    #[test]
    fn test_gpu_bfs_disconnected() {
        let (rp, ci) = build_csr(4, &[(0, 1), (2, 3)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], usize::MAX);
        assert_eq!(dist[3], usize::MAX);
    }

    #[test]
    fn test_gpu_bfs_single_node() {
        let rp = vec![0usize, 0];
        let ci: Vec<usize> = vec![];
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0]);
    }

    #[test]
    fn test_gpu_bfs_invalid_source() {
        let (rp, ci) = build_csr(4, &[(0, 1)]);
        assert!(gpu_bfs(&rp, &ci, 10, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_gpu_bfs_tree() {
        let (rp, ci) = build_csr(7, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_gpu_bfs_star_graph() {
        let edges: Vec<(usize, usize)> = (1..=5).map(|i| (0, i)).collect();
        let (rp, ci) = build_csr(6, &edges);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        for i in 1..=5 {
            assert_eq!(dist[i], 1);
        }
    }

    #[test]
    fn test_gpu_bfs_cycle() {
        let (rp, ci) = build_csr(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], 2);
        // 0-3 is a direct edge in undirected sense
        assert_eq!(dist[3], 1);
    }

    #[test]
    fn test_gpu_sssp_shortest_paths() {
        // Triangle: 0->1 (1), 0->2 (4), 1->2 (2) → dist[2]=3
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0), (0, 2, 4.0), (1, 2, 2.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert!(
            (dist[2] - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            dist[2]
        );
    }

    #[test]
    fn test_gpu_sssp_negative_weight_detection() {
        // Negative cycle: 0->1 (1), 1->0 (-2)
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0), (1, 0, -2.0), (0, 2, 5.0)]);
        assert!(gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_gpu_sssp_unreachable() {
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        assert_eq!(dist[2], f64::INFINITY);
    }

    #[test]
    fn test_gpu_sssp_path_graph() {
        let (rp, ci, w) = build_csr_directed(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        for i in 0..4usize {
            assert!(
                (dist[i] - i as f64).abs() < 1e-10,
                "dist[{}]={}",
                i,
                dist[i]
            );
        }
    }

    #[test]
    fn test_delta_stepping_matches_dijkstra() {
        let adj = vec![
            vec![(1usize, 2.0f64), (2, 6.0)],
            vec![(3usize, 1.0f64), (2, 3.0)],
            vec![(4usize, 1.0f64)],
            vec![(4usize, 5.0f64)],
            vec![],
        ];
        let delta_dist = gpu_sssp_delta_stepping(&adj, 0, 2.0, &GpuBfsConfig::default())
            .expect("delta stepping failed");
        let ref_dist = dijkstra_ref(&adj, 0);
        for i in 0..5 {
            if ref_dist[i] == f64::INFINITY {
                assert_eq!(delta_dist[i], f64::INFINITY);
            } else {
                assert!(
                    (ref_dist[i] - delta_dist[i]).abs() < 1e-9,
                    "node {}: ref={}, delta={}",
                    i,
                    ref_dist[i],
                    delta_dist[i]
                );
            }
        }
    }

    #[test]
    fn test_delta_stepping_negative_weight_error() {
        let adj = vec![vec![(1usize, -1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 0, 1.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_invalid_source() {
        let adj = vec![vec![(1usize, 1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 5, 1.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_invalid_delta() {
        let adj = vec![vec![(1usize, 1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 0, -1.0, &GpuBfsConfig::default()).is_err());
        assert!(gpu_sssp_delta_stepping(&adj, 0, 0.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_disconnected() {
        let adj = vec![
            vec![(1usize, 1.0f64)],
            vec![],
            vec![(3usize, 2.0f64)],
            vec![],
        ];
        let dist = gpu_sssp_delta_stepping(&adj, 0, 1.0, &GpuBfsConfig::default()).expect("failed");
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert_eq!(dist[2], f64::INFINITY);
        assert_eq!(dist[3], f64::INFINITY);
    }

    #[test]
    fn test_parallel_bfs_atomic_matches_bfs() {
        let (rp, ci) = build_csr(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let bfs_dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        let atomic_dist = parallel_bfs_atomic(&rp, &ci, 0);
        assert_eq!(bfs_dist, atomic_dist);
    }
}
