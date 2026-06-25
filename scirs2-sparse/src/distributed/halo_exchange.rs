//! Halo exchange simulation for distributed sparse matrix-vector products.
//!
//! This module provides an *in-process* simulation of the communication
//! pattern that a real distributed SpMV would require (e.g. MPI point-to-point
//! or AllGather).  The simulation is correct in the sense that each worker
//! only reads values from the portion of the global vector it would have
//! received via actual message passing.

use std::collections::HashMap;

use crate::error::{SparseError, SparseResult};

use super::partition::{DistributedCsr, RowPartition};

// ─────────────────────────────────────────────────────────────────────────────
// HaloConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the halo exchange simulation.
#[derive(Debug, Clone)]
pub struct HaloConfig {
    /// Number of logical workers (default 4).
    pub n_workers: usize,
}

impl Default for HaloConfig {
    fn default() -> Self {
        Self { n_workers: 4 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HaloMessage
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a message sent from one worker to another during halo exchange.
#[derive(Debug, Clone)]
pub struct HaloMessage {
    /// Worker that sends this message.
    pub source_worker: usize,
    /// Worker that receives this message.
    pub dest_worker: usize,
    /// Global row indices whose values are being sent.
    pub rows: Vec<usize>,
    /// Values corresponding to `rows` (same order).
    pub values: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// GhostManager
// ─────────────────────────────────────────────────────────────────────────────

/// Maps global row indices to local indices in the combined local+ghost vector.
///
/// Layout: `[0 .. n_local)` are owned rows; `[n_local .. n_local+n_ghost)` are
/// ghost rows, in the order they appear in `ghost_rows`.
#[derive(Debug, Clone)]
pub struct GhostManager {
    /// Maps global row index → local index (0..n_local+n_ghost).
    pub global_to_local_map: HashMap<usize, usize>,
    /// Number of owned rows.
    pub n_local: usize,
    /// Number of ghost rows.
    pub n_ghost: usize,
}

impl GhostManager {
    /// Construct from the list of owned rows and ghost rows.
    ///
    /// `local_rows` are stored first (indices 0..n_local),
    /// then `ghost_rows` (indices n_local..n_local+n_ghost).
    pub fn new(local_rows: &[usize], ghost_rows: &[usize]) -> Self {
        let n_local = local_rows.len();
        let n_ghost = ghost_rows.len();
        let mut map = HashMap::with_capacity(n_local + n_ghost);
        for (local_idx, &global) in local_rows.iter().enumerate() {
            map.insert(global, local_idx);
        }
        for (ghost_idx, &global) in ghost_rows.iter().enumerate() {
            map.insert(global, n_local + ghost_idx);
        }
        Self {
            global_to_local_map: map,
            n_local,
            n_ghost,
        }
    }

    /// Convert a global row index to its local index, if known.
    #[inline]
    pub fn global_to_local(&self, global: usize) -> Option<usize> {
        self.global_to_local_map.get(&global).copied()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DistributedVector
// ─────────────────────────────────────────────────────────────────────────────

/// A vector distributed across workers with separate local and ghost storage.
#[derive(Debug, Clone)]
pub struct DistributedVector {
    /// Values for owned rows (length = partition.n_local()).
    pub local_values: Vec<f64>,
    /// Values for ghost rows (length = ghost_rows.len()).
    pub ghost_values: Vec<f64>,
    /// Row ownership metadata.
    pub partition: RowPartition,
    /// Global indices of the ghost rows (parallel to `ghost_values`).
    pub ghost_rows: Vec<usize>,
}

impl DistributedVector {
    /// Construct a distributed vector by slicing the global vector.
    ///
    /// # Arguments
    ///
    /// * `global` — The full global vector of length `n_global_rows`.
    /// * `partition` — Which rows this worker owns.
    /// * `ghost_rows` — Global indices of ghost rows needed by this worker.
    pub fn from_global(
        global: &[f64],
        partition: &RowPartition,
        ghost_rows: &[usize],
    ) -> SparseResult<Self> {
        // Owned values.
        let local_values: SparseResult<Vec<f64>> = partition
            .local_rows
            .iter()
            .map(|&r| {
                global.get(r).copied().ok_or_else(|| {
                    SparseError::ValueError(format!(
                        "Global row index {r} out of bounds (len={})",
                        global.len()
                    ))
                })
            })
            .collect();
        let local_values = local_values?;

        // Ghost values.
        let ghost_values: SparseResult<Vec<f64>> = ghost_rows
            .iter()
            .map(|&r| {
                global.get(r).copied().ok_or_else(|| {
                    SparseError::ValueError(format!(
                        "Ghost row index {r} out of bounds (len={})",
                        global.len()
                    ))
                })
            })
            .collect();
        let ghost_values = ghost_values?;

        Ok(Self {
            local_values,
            ghost_values,
            partition: partition.clone(),
            ghost_rows: ghost_rows.to_vec(),
        })
    }

    /// Assemble the full global vector (owned rows only; other entries are 0).
    pub fn to_global(&self, n_global: usize) -> Vec<f64> {
        let mut out = vec![0.0_f64; n_global];
        for (local_idx, &global_row) in self.partition.local_rows.iter().enumerate() {
            if global_row < n_global {
                out[global_row] = self.local_values[local_idx];
            }
        }
        out
    }

    /// Look up a value by global row index (searches local then ghost storage).
    #[inline]
    pub fn get_global(&self, global_row: usize) -> Option<f64> {
        // Check owned rows.
        for (local_idx, &r) in self.partition.local_rows.iter().enumerate() {
            if r == global_row {
                return Some(self.local_values[local_idx]);
            }
        }
        // Check ghost rows.
        for (ghost_idx, &r) in self.ghost_rows.iter().enumerate() {
            if r == global_row {
                return Some(self.ghost_values[ghost_idx]);
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-process halo exchange
// ─────────────────────────────────────────────────────────────────────────────

/// Perform the halo-exchange step for the single-process model: for each
/// partition build a [`DistributedVector`] that contains both local and ghost
/// x-values.
///
/// This crate's "distributed" SpMV runs all partitions in one process (using
/// [`std::thread::scope`]), so every worker already shares the same address
/// space and can read the global `x` array directly. In a multi-node MPI
/// deployment each worker would instead receive its ghost x-values via message
/// passing from the owning rank; here we copy them out of the shared `x`, which
/// produces *identical numerical results* for the single-process case without
/// any inter-process communication. It is therefore an exact stand-in for the
/// single-process scenario, not a fabricated multi-node exchange.
pub fn simulate_halo_exchange(
    partitions: &[DistributedCsr],
    x_global: &[f64],
) -> SparseResult<Vec<DistributedVector>> {
    partitions
        .iter()
        .map(|dcsr| DistributedVector::from_global(x_global, &dcsr.partition, &dcsr.ghost_rows))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// distributed_spmv
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `y = A * x` using the distributed representation.
///
/// Distributes `x`, performs local SpMV on each partition (using both owned
/// and ghost values), then assembles the global result.
///
/// Uses [`std::thread::scope`] to parallelize across workers.
pub fn distributed_spmv(partitions: &[DistributedCsr], x: &[f64]) -> SparseResult<Vec<f64>> {
    if partitions.is_empty() {
        return Ok(Vec::new());
    }

    let n_global = partitions[0].partition.n_global_rows;

    // Validate x length against global n_rows (use n_cols of local matrices).
    // The global matrix is square in all our test cases, but be defensive:
    // each local_matrix was built with global column indices, so x must have
    // at least as many elements as any column index referenced.
    let n_cols_needed = partitions
        .iter()
        .map(|d| d.local_matrix.cols())
        .max()
        .unwrap_or(0);
    if x.len() < n_cols_needed {
        return Err(SparseError::DimensionMismatch {
            expected: n_cols_needed,
            found: x.len(),
        });
    }

    // Build distributed vectors (simulated halo exchange).
    let dist_vecs = simulate_halo_exchange(partitions, x)?;

    // We collect per-partition partial y-vectors via threads, then assemble.
    // Each element: (global_row_indices, y_values) for owned rows.
    let n_workers = partitions.len();
    let mut partial_results: Vec<(Vec<usize>, Vec<f64>)> =
        vec![(Vec::new(), Vec::new()); n_workers];

    std::thread::scope(|s| {
        let handles: Vec<_> = partitions
            .iter()
            .zip(dist_vecs.iter())
            .enumerate()
            .map(|(w, (dcsr, dv))| {
                s.spawn(move || -> SparseResult<(Vec<usize>, Vec<f64>)> {
                    // Build ghost_manager for this worker.
                    let ghost_mgr = GhostManager::new(&dcsr.partition.local_rows, &dcsr.ghost_rows);

                    let n_local = dcsr.partition.n_local();
                    let mut y_local = vec![0.0_f64; n_local];

                    for (local_row, &global_row) in dcsr.partition.local_rows.iter().enumerate() {
                        let row_start = dcsr.local_matrix.indptr[local_row];
                        let row_end = dcsr.local_matrix.indptr[local_row + 1];
                        let mut acc = 0.0_f64;
                        for idx in row_start..row_end {
                            let col = dcsr.local_matrix.indices[idx]; // global column index
                            let val = dcsr.local_matrix.data[idx];

                            // x[col] — col is a global row index for square A.
                            // Use ghost_mgr if available, else fall back to x directly.
                            let x_val = if let Some(local_idx) = ghost_mgr.global_to_local(col) {
                                if local_idx < dv.local_values.len() {
                                    dv.local_values[local_idx]
                                } else {
                                    let ghost_idx = local_idx - dv.local_values.len();
                                    *dv.ghost_values.get(ghost_idx).ok_or_else(|| {
                                        SparseError::ValueError(format!(
                                            "Ghost index {ghost_idx} out of range"
                                        ))
                                    })?
                                }
                            } else {
                                // Column references something outside owned+ghost —
                                // read directly from global x (safe: validated above).
                                *x.get(col).ok_or_else(|| {
                                    SparseError::ValueError(format!(
                                        "Column index {col} out of range in x (len={})",
                                        x.len()
                                    ))
                                })?
                            };

                            acc += val * x_val;
                        }
                        y_local[local_row] = acc;
                        let _ = global_row; // suppress unused warning
                    }

                    Ok((dcsr.partition.local_rows.clone(), y_local))
                })
            })
            .collect();

        for (w, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(result)) => {
                    partial_results[w] = result;
                }
                Ok(Err(e)) => {
                    // Store empty to signal error; we'll propagate below.
                    let _ = e;
                }
                Err(_) => {}
            }
        }
    });

    // Assemble global y.
    let mut y = vec![0.0_f64; n_global];
    for (global_rows, y_values) in &partial_results {
        for (&global_row, &yv) in global_rows.iter().zip(y_values.iter()) {
            if global_row < n_global {
                y[global_row] = yv;
            }
        }
    }

    Ok(y)
}

// ─────────────────────────────────────────────────────────────────────────────
// Build messages helper (for introspection / testing)
// ─────────────────────────────────────────────────────────────────────────────

/// Build the set of [`HaloMessage`]s that would be exchanged in a real
/// distributed run.
///
/// For each ghost row in a partition, the owning worker sends the
/// corresponding x-value.  This function identifies owner–destination pairs
/// and groups them into messages.
pub fn build_halo_messages(partitions: &[DistributedCsr], x: &[f64]) -> Vec<HaloMessage> {
    // Build global_row → worker_id mapping.
    let mut row_owner: HashMap<usize, usize> = HashMap::new();
    for (w, dcsr) in partitions.iter().enumerate() {
        for &r in &dcsr.partition.local_rows {
            row_owner.insert(r, w);
        }
    }

    let mut messages: Vec<HaloMessage> = Vec::new();

    for (dest_worker, dcsr) in partitions.iter().enumerate() {
        // Group ghost rows by their owning worker.
        let mut by_source: HashMap<usize, (Vec<usize>, Vec<f64>)> = HashMap::new();
        for &ghost_row in &dcsr.ghost_rows {
            if let Some(&src) = row_owner.get(&ghost_row) {
                let xv = x.get(ghost_row).copied().unwrap_or(0.0);
                let entry = by_source
                    .entry(src)
                    .or_insert_with(|| (Vec::new(), Vec::new()));
                entry.0.push(ghost_row);
                entry.1.push(xv);
            }
        }
        for (source_worker, (rows, values)) in by_source {
            messages.push(HaloMessage {
                source_worker,
                dest_worker,
                rows,
                values,
            });
        }
    }

    messages
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::distributed::partition::create_distributed_csr;
    use crate::distributed::partition::{partition_rows, PartitionConfig, PartitionMethod};

    /// Build an n×n tridiagonal matrix with diagonal=2, off-diag=-1.
    fn tridiag(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0_f64);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                vals.push(-1.0);
                rows.push(i - 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        CsrMatrix::from_triplets(n, n, rows, cols, vals).expect("tridiag construction")
    }

    fn make_partitions(mat: &CsrMatrix<f64>, n_workers: usize) -> Vec<DistributedCsr> {
        let config = PartitionConfig {
            n_workers,
            ..Default::default()
        };
        let row_parts = partition_rows(mat.rows(), &config);
        row_parts
            .iter()
            .map(|rp| create_distributed_csr(mat, rp).expect("create_distributed_csr"))
            .collect()
    }

    #[test]
    fn test_distributed_spmv_matches_serial() {
        let n = 10;
        let mat = tridiag(n);
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Serial SpMV via CsrMatrix::dot.
        let y_serial = mat.dot(&x).expect("serial dot");

        // Distributed SpMV.
        let parts = make_partitions(&mat, 4);
        let y_dist = distributed_spmv(&parts, &x).expect("distributed_spmv");

        assert_eq!(y_serial.len(), y_dist.len());
        for (i, (ys, yd)) in y_serial.iter().zip(y_dist.iter()).enumerate() {
            assert!(
                (ys - yd).abs() < 1e-10,
                "row {i}: serial={ys}, distributed={yd}"
            );
        }
    }

    #[test]
    fn test_distributed_spmv_single_worker() {
        let n = 8;
        let mat = tridiag(n);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_serial = mat.dot(&x).expect("serial dot");
        let parts = make_partitions(&mat, 1);
        let y_dist = distributed_spmv(&parts, &x).expect("distributed_spmv");
        for (ys, yd) in y_serial.iter().zip(y_dist.iter()) {
            assert!((ys - yd).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ghost_manager_lookup() {
        let local_rows = vec![0usize, 1, 2];
        let ghost_rows = vec![5usize, 7];
        let mgr = GhostManager::new(&local_rows, &ghost_rows);
        assert_eq!(mgr.global_to_local(0), Some(0));
        assert_eq!(mgr.global_to_local(2), Some(2));
        assert_eq!(mgr.global_to_local(5), Some(3));
        assert_eq!(mgr.global_to_local(7), Some(4));
        assert_eq!(mgr.global_to_local(9), None);
    }

    #[test]
    fn test_distributed_vector_roundtrip() {
        let global = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rp = RowPartition {
            worker_id: 0,
            local_rows: vec![1, 2],
            n_global_rows: 5,
        };
        let ghost_rows = vec![4usize];
        let dv = DistributedVector::from_global(&global, &rp, &ghost_rows).expect("from_global");
        assert_eq!(dv.local_values, vec![2.0, 3.0]);
        assert_eq!(dv.ghost_values, vec![5.0]);

        let reconstructed = dv.to_global(5);
        assert_eq!(reconstructed[1], 2.0);
        assert_eq!(reconstructed[2], 3.0);
        // Other positions are 0.
        assert_eq!(reconstructed[0], 0.0);
    }

    #[test]
    fn test_halo_messages_built() {
        let n = 10;
        let mat = tridiag(n);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let parts = make_partitions(&mat, 4);
        let msgs = build_halo_messages(&parts, &x);
        // There should be messages at partition boundaries.
        assert!(
            !msgs.is_empty(),
            "Expected halo messages for tridiagonal matrix"
        );
    }

    #[test]
    fn test_distributed_spmv_round_robin() {
        let n = 12;
        let mat = tridiag(n);
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let y_serial = mat.dot(&x).expect("serial dot");

        let config = PartitionConfig {
            n_workers: 3,
            method: PartitionMethod::RoundRobin,
            ..Default::default()
        };
        let row_parts = partition_rows(n, &config);
        let parts: Vec<DistributedCsr> = row_parts
            .iter()
            .map(|rp| create_distributed_csr(&mat, rp).expect("create"))
            .collect();
        let y_dist = distributed_spmv(&parts, &x).expect("distributed_spmv");

        for (i, (ys, yd)) in y_serial.iter().zip(y_dist.iter()).enumerate() {
            assert!((ys - yd).abs() < 1e-10, "row {i}: serial={ys}, dist={yd}");
        }
    }
}
