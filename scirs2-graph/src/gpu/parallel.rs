//! CPU-parallel graph algorithm helpers.
//!
//! These routines provide the `CpuParallel` backend implementations for BFS and SSSP.
//! They use atomic operations for thread-safe concurrent frontier/distance updates
//! and are designed to mirror the GPU shader semantics closely.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::error::{GraphError, Result};

/// Parallel level-synchronous BFS using `AtomicUsize` for distance flags.
///
/// Uses compare-and-swap on each neighbor: only the first thread to win the CAS
/// claims that neighbor, preventing duplicate frontier entries.
/// Returns distances in the same format as the GPU path (`usize::MAX` = unreachable).
pub(crate) fn parallel_bfs_atomic(
    row_ptr: &[usize],
    col_idx: &[usize],
    source: usize,
) -> Vec<usize> {
    if row_ptr.len() < 2 {
        return vec![];
    }
    let n = row_ptr.len() - 1;
    if source >= n {
        return vec![usize::MAX; n];
    }

    let dist: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
    dist[source].store(0, Ordering::Relaxed);

    let mut frontier = vec![source];

    while !frontier.is_empty() {
        let mut next: Vec<usize> = Vec::new();
        for &v in &frontier {
            let d_v = dist[v].load(Ordering::Relaxed);
            let start = row_ptr[v];
            let end = row_ptr[v + 1];
            for &nb in &col_idx[start..end] {
                if dist[nb]
                    .compare_exchange(usize::MAX, d_v + 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    next.push(nb);
                }
            }
        }
        frontier = next;
    }

    dist.into_iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect()
}

/// Parallel edge-relaxation Bellman-Ford SSSP using the f32-bits `AtomicU32` trick.
///
/// # Invariants
/// - All edge weights must be non-negative and finite (checked on entry).
/// - Uses `f32::to_bits()` / `u32::from_bits()` IEEE 754 atomicMin pattern.
///
/// # Errors
/// Returns `GraphError::InvalidParameter` if any weight is negative or non-finite,
/// or if `source` is out of range.
pub(crate) fn parallel_bellman_ford_atomic(
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f32],
    source: usize,
) -> Result<Vec<f32>> {
    if row_ptr.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "row_ptr".to_string(),
            value: format!("len={}", row_ptr.len()),
            expected: "at least 2 elements (n+1)".to_string(),
            context: "parallel_bellman_ford_atomic".to_string(),
        });
    }
    let n = row_ptr.len() - 1;
    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{source}"),
            expected: format!("0..{n}"),
            context: "parallel_bellman_ford_atomic".to_string(),
        });
    }

    // Reject negative or non-finite weights upfront (atomicMin trick requires non-negative finite)
    for (idx, &w) in weights.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            return Err(GraphError::InvalidParameter {
                param: "weights".to_string(),
                value: format!("weight[{idx}]={w}"),
                expected: "non-negative finite weights for parallel Bellman-Ford".to_string(),
                context: "parallel_bellman_ford_atomic".to_string(),
            });
        }
    }

    // Initialise distances: source=0, all others=infinity (as u32 bits)
    let inf_bits = f32::INFINITY.to_bits();
    let dist: Vec<AtomicU32> = (0..n)
        .map(|i| {
            if i == source {
                AtomicU32::new(0u32.to_le()) // 0.0f32.to_bits() == 0
            } else {
                AtomicU32::new(inf_bits)
            }
        })
        .collect();

    // n-1 relaxation passes (standard Bellman-Ford)
    for _ in 0..(n.saturating_sub(1)) {
        let mut changed = false;
        for u in 0..n {
            let d_u_bits = dist[u].load(Ordering::Relaxed);
            let d_u = f32::from_bits(d_u_bits);
            if !d_u.is_finite() {
                continue;
            }
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_idx[idx];
                let w = weights[idx];
                let nd = d_u + w;
                let nd_bits = nd.to_bits();
                // atomicMin via fetch_update: works correctly for non-negative IEEE 754 f32
                let prev = dist[v].fetch_min(nd_bits, Ordering::Relaxed);
                if nd_bits < prev {
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    Ok(dist
        .into_iter()
        .map(|a| f32::from_bits(a.load(Ordering::Relaxed)))
        .collect())
}
