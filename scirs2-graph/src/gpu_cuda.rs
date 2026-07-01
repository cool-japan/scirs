#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only CUDA acceleration for sparse graph
//! linear algebra — specifically a CSR sparse matrix-vector product (SpMV) on a
//! graph's adjacency matrix.
//!
//! This module is compiled only when the `cuda` feature is enabled. Even then it
//! is *runtime-probed*: [`cuda_is_available`] returns `false` on any platform
//! without an NVIDIA CUDA device (for example macOS), so callers can compile
//! everywhere and dispatch to the CPU path
//! ([`crate::compressed::CsrGraph::spmv`]) when no device is present. Everything
//! here is `f64`-native — there is no silent `f32` downcast at the boundary.
//!
//! # Library-call pattern (vs. custom-kernel pattern)
//!
//! Like `scirs2-interpolate`'s CUDA path — and unlike `scirs2-symbolic`'s, which
//! *emits custom kernels* through the `oxicuda-ptx` builder — this module follows
//! the **library-call pattern**: it hands the sparse matrix-vector multiply off
//! to the pre-built, battle-tested `oxicuda-sparse` *library* crate (a pure-Rust
//! cuSPARSE equivalent). No bespoke kernels are authored here; the
//! adjacency-matrix CSR is uploaded and `oxicuda_sparse::ops::spmv` does the work.
//!
//! # What this accelerates, and where it generalizes
//!
//! [`cuda_spmv_csr`] computes `y = A · x`, where `A` is the graph's adjacency
//! matrix in CSR form (`row_offsets` / `col_indices` / `values`) and `x` is a
//! dense per-node vector. `y = A · x` is *the* fundamental graph-linear-algebra
//! primitive: PageRank is iterated SpMV, and a single BFS / SSSP frontier-
//! expansion step is an SpMV over an appropriate semiring (boolean OR-AND for
//! BFS, min-plus for SSSP). This slice wires the **plus-times** (numeric) SpMV;
//! the semiring variants are a natural future extension once `oxicuda-sparse`
//! exposes semiring SpMV.

use crate::error::{GraphError, Result as GraphResult};
use oxicuda_sparse::ops::{spmv, SpMVAlgo};
use oxicuda_sparse::{CsrMatrix, SparseHandle};

/// Returns `true` only when an NVIDIA CUDA device is present and the driver
/// initialises successfully.
///
/// This never panics: on a non-NVIDIA platform such as macOS the CUDA driver
/// fails to initialise (or reports zero devices) and this returns `false`,
/// allowing callers to fall back to the CPU SpMV in
/// [`crate::compressed::CsrGraph::spmv`].
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Maps an `oxicuda-sparse` error onto the crate's honest `ComputationError`.
fn sparse_err(e: oxicuda_sparse::error::SparseError) -> GraphError {
    GraphError::ComputationError(format!("oxicuda-sparse: {e}"))
}

/// Maps an `oxicuda` CUDA driver/memory error onto `ComputationError`.
fn cuda_err(e: oxicuda_driver::CudaError) -> GraphError {
    GraphError::ComputationError(format!("oxicuda CUDA driver: {e}"))
}

/// Initialises the CUDA driver, selects device 0, and builds a shared context.
///
/// Returns a [`ComputationError`](GraphError::ComputationError) — never a
/// fabricated success — when CUDA is unavailable or no device is present.
fn build_context() -> GraphResult<std::sync::Arc<oxicuda_driver::Context>> {
    oxicuda_driver::init()
        .map_err(|e| GraphError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| GraphError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(GraphError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    Ok(std::sync::Arc::new(
        oxicuda_driver::Context::new(&dev).map_err(cuda_err)?,
    ))
}

/// Computes the sparse matrix-vector product `y = A · x` on the GPU, where `A`
/// is a graph adjacency matrix in CSR (Compressed Sparse Row) form.
///
/// This is the GPU companion to the CPU [`crate::compressed::CsrGraph::spmv`].
/// `A` is `n_rows × n_cols`; for a graph adjacency it is square with
/// `n_rows == n_cols == num_nodes`. The CSR arrays follow oxicuda-sparse's
/// `i32` index convention:
///
/// * `row_offsets` — length `n_rows + 1`, `row_offsets[0] == 0`,
///   `row_offsets[n_rows] == nnz`, non-decreasing. (This is
///   `CsrGraph::row_ptr()` cast from `usize` to `i32`.)
/// * `col_indices` — length `nnz`, each in `0..n_cols`. (This is
///   `CsrGraph::col_indices()` cast from `usize` to `i32`.)
/// * `values` — length `nnz`, the edge weights (`CsrGraph::values()`).
/// * `x` — dense input vector of length `n_cols`.
///
/// Returns `y` of length `n_rows`. The multiply is routed through
/// `oxicuda_sparse::ops::spmv` with `alpha = 1.0`, `beta = 0.0` and
/// [`SpMVAlgo::Adaptive`] (which picks a scalar- or vector-per-row kernel from
/// the matrix's average nnz/row).
///
/// `y = A · x` is the fundamental graph-linear-algebra primitive: PageRank is
/// iterated SpMV, and a single BFS / SSSP frontier-expansion step is an SpMV
/// over a boolean / min-plus semiring respectively — natural future extensions.
///
/// # Errors
///
/// Returns [`ComputationError`](GraphError::ComputationError) — never a
/// fabricated success — when CUDA is unavailable, when no NVIDIA device is
/// present, or when oxicuda-sparse reports a failure. Shape inconsistencies are
/// reported as [`InvalidGraph`](GraphError::InvalidGraph) *before* any device
/// work, so they are detectable without a GPU.
pub fn cuda_spmv_csr(
    row_offsets: &[i32],
    col_indices: &[i32],
    values: &[f64],
    n_rows: usize,
    n_cols: usize,
    x: &[f64],
) -> GraphResult<Vec<f64>> {
    // ---- Host-side validation (runs without a GPU) ----
    if n_rows == 0 {
        // A 0×n matrix maps any x to an empty y.
        return Ok(Vec::new());
    }
    if row_offsets.len() != n_rows + 1 {
        return Err(GraphError::InvalidGraph(format!(
            "cuda_spmv_csr: row_offsets length {} does not match n_rows + 1 = {}",
            row_offsets.len(),
            n_rows + 1
        )));
    }
    if col_indices.len() != values.len() {
        return Err(GraphError::InvalidGraph(format!(
            "cuda_spmv_csr: col_indices length {} != values length {}",
            col_indices.len(),
            values.len()
        )));
    }
    if x.len() != n_cols {
        return Err(GraphError::InvalidGraph(format!(
            "cuda_spmv_csr: x length {} != n_cols {}",
            x.len(),
            n_cols
        )));
    }
    let nnz = values.len();
    let last = *row_offsets.last().unwrap_or(&0);
    if i64::from(last) != nnz as i64 {
        return Err(GraphError::InvalidGraph(format!(
            "cuda_spmv_csr: row_offsets last element {last} != nnz {nnz}"
        )));
    }

    // An adjacency with no stored edges sends every node's value to 0.
    if nnz == 0 {
        return Ok(vec![0.0f64; n_rows]);
    }

    // ---- GPU path ----
    let ctx = build_context()?;
    let handle = SparseHandle::new(&ctx).map_err(sparse_err)?;

    // Upload the CSR adjacency. CsrMatrix::from_host validates structure and
    // uploads row_ptr / col_idx / values to device memory.
    let csr = CsrMatrix::<f64>::from_host(
        n_rows as u32,
        n_cols as u32,
        row_offsets,
        col_indices,
        values,
    )
    .map_err(sparse_err)?;

    // Upload x, and a zero-initialised y. spmv computes y = alpha*A*x + beta*y
    // and reads y even when beta = 0, so y must not be uninitialised / NaN.
    let d_x = oxicuda_memory::DeviceBuffer::from_host(x).map_err(cuda_err)?;
    let d_y = oxicuda_memory::DeviceBuffer::from_host(&vec![0.0f64; n_rows]).map_err(cuda_err)?;

    // y = 1.0 * A * x + 0.0 * y
    spmv::<f64>(
        &handle,
        SpMVAlgo::Adaptive,
        1.0,
        &csr,
        d_x.as_device_ptr(),
        0.0,
        d_y.as_device_ptr(),
    )
    .map_err(sparse_err)?;

    let mut y = vec![0.0f64; n_rows];
    d_y.copy_to_host(&mut y).map_err(cuda_err)?;
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Always-run tests: host-side validation, no GPU required ----

    #[test]
    fn shape_mismatch_is_detected_without_gpu() {
        // row_offsets length must be n_rows + 1 (= 4); here it is too short.
        let res = cuda_spmv_csr(&[0, 1], &[0], &[1.0], 3, 3, &[1.0, 2.0, 3.0]);
        assert!(res.is_err());
    }

    #[test]
    fn x_length_mismatch_is_detected_without_gpu() {
        // Consistent CSR (1 nnz) but x has length 2 while n_cols = 3.
        let res = cuda_spmv_csr(&[0, 1, 1, 1], &[0], &[1.0], 3, 3, &[1.0, 2.0]);
        assert!(res.is_err());
    }

    #[test]
    fn empty_matrix_is_ok_without_gpu() {
        let y = cuda_spmv_csr(&[], &[], &[], 0, 0, &[]).expect("empty spmv should be Ok");
        assert!(y.is_empty());
    }

    #[test]
    fn no_edges_yields_zero_vector_without_gpu() {
        // 3 nodes, no edges: row_offsets all zero, nnz = 0 => y = [0, 0, 0].
        let y = cuda_spmv_csr(&[0, 0, 0, 0], &[], &[], 3, 3, &[1.0, 2.0, 3.0])
            .expect("no-edge spmv should be Ok");
        assert_eq!(y, vec![0.0, 0.0, 0.0]);
    }

    // ---- Skip-on-no-GPU smoke tests: real device SpMV vs CPU reference ----

    #[test]
    fn cuda_spmv_path_graph_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        // Undirected 3-node path graph 0 - 1 - 2 with unit weights.
        // Adjacency A (symmetric):
        //   row 0: col 1
        //   row 1: cols 0, 2
        //   row 2: col 1
        // CSR: row_ptr = [0, 1, 3, 4], col = [1, 0, 2, 1], val = [1, 1, 1, 1]
        use crate::compressed::CsrGraph;
        let row_ptr_us = vec![0usize, 1, 3, 4];
        let col_us = vec![1usize, 0, 2, 1];
        let vals = vec![1.0f64, 1.0, 1.0, 1.0];

        let g = CsrGraph::from_raw(3, row_ptr_us.clone(), col_us.clone(), vals.clone(), false)
            .expect("from_raw");
        let x = vec![1.0f64, 2.0, 3.0];
        let cpu = g.spmv(&x).expect("cpu spmv");
        // Hand-check: y0 = x1 = 2; y1 = x0 + x2 = 4; y2 = x1 = 2.
        assert_eq!(cpu, vec![2.0, 4.0, 2.0]);

        let row_ptr_i: Vec<i32> = row_ptr_us.iter().map(|&v| v as i32).collect();
        let col_i: Vec<i32> = col_us.iter().map(|&v| v as i32).collect();
        let gpu = cuda_spmv_csr(&row_ptr_i, &col_i, &vals, 3, 3, &x).expect("gpu spmv");

        let max_diff = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(gv, cv)| (gv - cv).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-9, "max abs diff {max_diff} exceeds 1e-9");
    }

    // ---- Vector-kernel SpMV: directed asymmetric graph, avg nnz/row ≥ 4.0 ----
    //
    // `SpMVAlgo::Adaptive` selects the SCALAR kernel when avg_nnz_per_row < 4.0
    // and the VECTOR (warp-per-row) kernel when avg_nnz_per_row >= 4.0
    // (VECTOR_THRESHOLD = 4.0 in oxicuda-sparse/src/ops/spmv.rs).
    //
    // The existing `cuda_spmv_path_graph_or_skip` test uses a 3-node path graph
    // (4 nnz / 3 rows = 1.33 avg) which routes to SCALAR.  This test uses an
    // 8-node DIRECTED, ASYMMETRIC graph with 37 nnz (avg = 37/8 = 4.625 ≥ 4.0)
    // so that `Adaptive` resolves to the VECTOR warp-shuffle-reduction kernel —
    // the kernel that was fixed for f64.  The adjacency is intentionally
    // non-symmetric (e.g. edge 0→1 has weight 0.5 while edge 1→0 has weight
    // 0.7), so any silent row/col transposition in the CSR upload would produce
    // wrong results.
    //
    // CSR layout (row → [col(weight), ...]):
    //   row 0 → [1(0.5),  2(1.5),  3(2.0),  4(0.3),  5(1.1)]  (5 edges)
    //   row 1 → [0(0.7),  3(1.2),  5(0.9),  6(2.3)]            (4 edges)
    //   row 2 → [0(0.4),  1(1.8),  4(0.6),  5(2.1),  7(1.4)]  (5 edges)
    //   row 3 → [0(0.8),  2(1.3),  6(0.2),  7(1.7)]            (4 edges)
    //   row 4 → [1(0.5),  2(0.9),  3(1.6),  5(2.2),  6(0.7)]  (5 edges)
    //   row 5 → [0(1.1),  3(0.4),  4(1.9),  7(0.6)]            (4 edges)
    //   row 6 → [1(0.3),  2(1.5),  3(0.8),  4(2.4),  5(1.0)]  (5 edges)
    //   row 7 → [0(0.6),  2(1.2),  4(0.8),  5(1.4),  6(0.9)]  (5 edges)
    //   total: 37 nnz, avg = 37/8 = 4.625 → VECTOR kernel selected
    #[test]
    fn cuda_spmv_vector_kernel_directed_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        use crate::compressed::CsrGraph;

        // 8-node directed graph — 37 nnz, avg nnz/row = 4.625 ≥ 4.0.
        let n: usize = 8;
        #[rustfmt::skip]
        let row_ptr_us: Vec<usize> = vec![0, 5, 9, 14, 18, 23, 27, 32, 37];
        #[rustfmt::skip]
        let col_us: Vec<usize> = vec![
            // row 0 (5 edges)
            1, 2, 3, 4, 5,
            // row 1 (4 edges)
            0, 3, 5, 6,
            // row 2 (5 edges)
            0, 1, 4, 5, 7,
            // row 3 (4 edges)
            0, 2, 6, 7,
            // row 4 (5 edges)
            1, 2, 3, 5, 6,
            // row 5 (4 edges)
            0, 3, 4, 7,
            // row 6 (5 edges)
            1, 2, 3, 4, 5,
            // row 7 (5 edges)
            0, 2, 4, 5, 6,
        ];
        #[rustfmt::skip]
        let vals: Vec<f64> = vec![
            // row 0
            0.5, 1.5, 2.0, 0.3, 1.1,
            // row 1 — note: (1,0)=0.7 ≠ (0,1)=0.5 → asymmetric
            0.7, 1.2, 0.9, 2.3,
            // row 2
            0.4, 1.8, 0.6, 2.1, 1.4,
            // row 3
            0.8, 1.3, 0.2, 1.7,
            // row 4
            0.5, 0.9, 1.6, 2.2, 0.7,
            // row 5
            1.1, 0.4, 1.9, 0.6,
            // row 6
            0.3, 1.5, 0.8, 2.4, 1.0,
            // row 7
            0.6, 1.2, 0.8, 1.4, 0.9,
        ];

        // Confirm avg nnz/row >= 4.0 so Adaptive selects the VECTOR kernel.
        let nnz = vals.len();
        let avg_nnz_per_row = nnz as f64 / n as f64;
        assert!(
            avg_nnz_per_row >= 4.0,
            "avg nnz/row {avg_nnz_per_row:.3} must be >= 4.0 to select the VECTOR kernel"
        );

        // CPU reference SpMV.
        let g = CsrGraph::from_raw(n, row_ptr_us.clone(), col_us.clone(), vals.clone(), true)
            .expect("from_raw directed graph");
        let x: Vec<f64> = (1..=n).map(|v| v as f64).collect(); // x = [1,2,3,4,5,6,7,8]
        let cpu = g.spmv(&x).expect("cpu spmv");

        // GPU SpMV via VECTOR kernel (avg nnz/row = 4.625 >= VECTOR_THRESHOLD = 4.0).
        let row_ptr_i: Vec<i32> = row_ptr_us.iter().map(|&v| v as i32).collect();
        let col_i: Vec<i32> = col_us.iter().map(|&v| v as i32).collect();
        let gpu = cuda_spmv_csr(&row_ptr_i, &col_i, &vals, n, n, &x)
            .expect("cuda_spmv_csr (vector kernel)");

        assert_eq!(gpu.len(), n, "output length must match n_rows");

        let max_diff = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(gv, cv)| (gv - cv).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-9,
            "VECTOR SpMV max abs diff {max_diff:.3e} exceeds 1e-9 \
             (avg_nnz_per_row={avg_nnz_per_row:.3}, nnz={nnz})"
        );
    }
}
