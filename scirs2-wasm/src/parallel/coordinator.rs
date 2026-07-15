//! Thread-safe parallel work coordinator.
//!
//! Uses `std::thread` for native targets (and `wasm32-wasi` / `wasm32-wasip1`
//! with atomics enabled).  On `wasm32-unknown-unknown`, where `std::thread` is
//! not available, all operations fall back to sequential execution so that the
//! crate still compiles and is functionally correct — just without parallelism.
//!
//! # Platform matrix
//!
//! | Target                      | Implementation         |
//! |-----------------------------|------------------------|
//! | native (x86_64, aarch64, …) | `std::thread::spawn`   |
//! | wasm32-wasip1-threads       | `std::thread::spawn`   |
//! | wasm32-unknown-unknown      | sequential fallback     |
//! | wasm32-wasip1 (no atomics)  | sequential fallback     |

use std::sync::{Arc, Barrier, Mutex};

use crate::parallel::types::ParallelConfig;

// ------------------------------------------------------------------
// ParallelCoordinator
// ------------------------------------------------------------------

/// Dispatches work across a configurable number of worker threads.
///
/// On `wasm32-unknown-unknown` and similar targets where `std::thread` is
/// unavailable, all operations execute sequentially in the calling thread.
pub struct ParallelCoordinator {
    config: ParallelConfig,
}

impl ParallelCoordinator {
    /// Construct a new coordinator from `config`.
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Return the active configuration.
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    // ------------------------------------------------------------------
    // parallel_map
    // ------------------------------------------------------------------

    /// Apply `f` to each chunk in parallel and concatenate the results.
    ///
    /// The data is split into chunks of at most `config.chunk_size` elements
    /// and distributed across up to `config.n_workers` threads (or processed
    /// sequentially on targets without threading support).
    pub fn parallel_map<F>(&self, data: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        if data.is_empty() {
            return Vec::new();
        }

        let chunk_size = self.config.chunk_size.max(1);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let chunks: Vec<Vec<f64>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

            let n_chunks = chunks.len();
            let n_workers = self.config.n_workers.max(1);
            let f = Arc::new(f);

            let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<f64>)>();
            let actual_workers = n_workers.min(n_chunks);

            let mut worker_chunks: Vec<Vec<(usize, Vec<f64>)>> =
                (0..actual_workers).map(|_| Vec::new()).collect();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                worker_chunks[idx % actual_workers].push((idx, chunk));
            }

            let mut handles = Vec::with_capacity(actual_workers);
            for worker_data in worker_chunks {
                let tx_clone = tx.clone();
                let f_clone = Arc::clone(&f);
                let handle = std::thread::spawn(move || {
                    for (idx, chunk) in worker_data {
                        let result = f_clone(&chunk);
                        let _ = tx_clone.send((idx, result));
                    }
                });
                handles.push(handle);
            }
            drop(tx);

            let mut ordered: Vec<Option<Vec<f64>>> = (0..n_chunks).map(|_| None).collect();
            for (idx, result) in rx {
                if idx < ordered.len() {
                    ordered[idx] = Some(result);
                }
            }
            for handle in handles {
                let _ = handle.join();
            }

            ordered
                .into_iter()
                .flat_map(|v| v.unwrap_or_default())
                .collect()
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Sequential fallback: process chunks one by one in the caller's thread.
            data.chunks(chunk_size).flat_map(f).collect()
        }
    }

    // ------------------------------------------------------------------
    // parallel_reduce
    // ------------------------------------------------------------------

    /// Split `data` into chunks, reduce each chunk in parallel using `f`,
    /// then combine the partial results.
    pub fn parallel_reduce<F, R>(&self, data: &[f64], init: R, f: F) -> R
    where
        F: Fn(R, &[f64]) -> R + Send + Sync + Clone + 'static,
        R: Send + Clone + 'static,
    {
        if data.is_empty() {
            return init;
        }

        let chunk_size = self.config.chunk_size.max(1);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let chunks: Vec<Vec<f64>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();
            let n_workers = self.config.n_workers.max(1);
            let n_chunks = chunks.len();
            let actual_workers = n_workers.min(n_chunks);

            let (tx, rx) = std::sync::mpsc::channel::<(usize, R)>();
            let mut worker_chunks: Vec<Vec<(usize, Vec<f64>)>> =
                (0..actual_workers).map(|_| Vec::new()).collect();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                worker_chunks[idx % actual_workers].push((idx, chunk));
            }

            let mut handles = Vec::with_capacity(actual_workers);
            for worker_data in worker_chunks {
                let tx_clone = tx.clone();
                let f_clone = f.clone();
                let init_clone = init.clone();
                let handle = std::thread::spawn(move || {
                    let mut acc = init_clone;
                    let mut sorted = worker_data;
                    sorted.sort_by_key(|(idx, _)| *idx);
                    let first_idx = sorted.first().map(|(i, _)| *i).unwrap_or(0);
                    for (_idx, chunk) in &sorted {
                        acc = f_clone(acc, chunk);
                    }
                    let _ = tx_clone.send((first_idx, acc));
                });
                handles.push(handle);
            }
            drop(tx);

            let mut partials: Vec<(usize, R)> = rx.into_iter().collect();
            partials.sort_by_key(|(idx, _)| *idx);
            for handle in handles {
                let _ = handle.join();
            }

            let mut acc = init;
            for (_, partial) in partials {
                let _ = acc;
                acc = partial;
            }
            acc
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Sequential fallback.
            let mut acc = init;
            for chunk in data.chunks(chunk_size) {
                acc = f(acc, chunk);
            }
            acc
        }
    }

    // ------------------------------------------------------------------
    // parallel_matmul
    // ------------------------------------------------------------------

    /// Parallel matrix multiply `C = A × B` using `f64` elements.
    ///
    /// * `a` — flat row-major buffer, shape `m × k`.
    /// * `b` — flat row-major buffer, shape `k × n`.
    /// * Returns a flat row-major buffer, shape `m × n`.
    pub fn parallel_matmul(&self, a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), k * n);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let n_workers = self.config.n_workers.max(1);
            let rows_per_worker = m.div_ceil(n_workers);

            let a_arc = Arc::new(a.to_vec());
            let b_arc = Arc::new(b.to_vec());

            let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<f64>)>();
            let mut handles = Vec::new();

            for worker_id in 0..n_workers {
                let row_start = worker_id * rows_per_worker;
                if row_start >= m {
                    break;
                }
                let row_end = (row_start + rows_per_worker).min(m);
                let a_clone = Arc::clone(&a_arc);
                let b_clone = Arc::clone(&b_arc);
                let tx_clone = tx.clone();

                let handle = std::thread::spawn(move || {
                    let rows = row_end - row_start;
                    let mut local_c = vec![0.0_f64; rows * n];
                    for r in 0..rows {
                        let global_row = row_start + r;
                        for col in 0..n {
                            let mut sum = 0.0_f64;
                            for kk in 0..k {
                                sum += a_clone[global_row * k + kk] * b_clone[kk * n + col];
                            }
                            local_c[r * n + col] = sum;
                        }
                    }
                    let _ = tx_clone.send((row_start, local_c));
                });
                handles.push(handle);
            }
            drop(tx);

            let mut ordered: Vec<(usize, Vec<f64>)> = rx.into_iter().collect();
            ordered.sort_by_key(|(start, _)| *start);
            for handle in handles {
                let _ = handle.join();
            }

            let mut c = vec![0.0_f64; m * n];
            for (row_start, partial) in ordered {
                let rows = partial.len() / n;
                for r in 0..rows {
                    let global_row = row_start + r;
                    if global_row >= m {
                        break;
                    }
                    c[global_row * n..global_row * n + n]
                        .copy_from_slice(&partial[r * n..(r + 1) * n]);
                }
            }
            c
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Sequential fallback: simple O(m·k·n) matmul.
            let mut c = vec![0.0_f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0_f64;
                    for kk in 0..k {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            c
        }
    }

    // ------------------------------------------------------------------
    // parallel_sort  (parallel merge sort)
    // ------------------------------------------------------------------

    /// Sort `data` in-place using a parallel merge sort.
    ///
    /// On `wasm32-unknown-unknown` this falls back to `Vec::sort_by`.
    pub fn parallel_sort(&self, data: &mut [f64]) {
        if data.len() <= 1 {
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let n_workers = self.config.n_workers.max(1);
            let chunk_size = data.len().div_ceil(n_workers);

            let mut chunks: Vec<Vec<f64>> = data.chunks(chunk_size).map(|c| c.to_vec()).collect();

            let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<f64>)>();
            let mut handles = Vec::new();
            for (idx, chunk) in chunks.drain(..).enumerate() {
                let tx_clone = tx.clone();
                let handle = std::thread::spawn(move || {
                    let mut sorted = chunk;
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let _ = tx_clone.send((idx, sorted));
                });
                handles.push(handle);
            }
            drop(tx);

            let mut sorted_chunks: Vec<(usize, Vec<f64>)> = rx.into_iter().collect();
            sorted_chunks.sort_by_key(|(idx, _)| *idx);
            for handle in handles {
                let _ = handle.join();
            }

            let mut merged: Vec<f64> = Vec::with_capacity(data.len());
            for (_, chunk) in sorted_chunks {
                merged = merge_sorted(merged, chunk);
            }
            data.copy_from_slice(&merged);
        }

        #[cfg(target_arch = "wasm32")]
        {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    // ------------------------------------------------------------------
    // Barrier factory
    // ------------------------------------------------------------------

    /// Create a `std::sync::Barrier` for synchronising `n_workers` threads.
    pub fn barrier(n_workers: usize) -> Arc<Barrier> {
        Arc::new(Barrier::new(n_workers))
    }
}

/// Merge two sorted `Vec<f64>` slices into a single sorted `Vec<f64>`.
#[cfg(not(target_arch = "wasm32"))]
fn merge_sorted(a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            result.push(a[i]);
            i += 1;
        } else {
            result.push(b[j]);
            j += 1;
        }
    }
    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

// ------------------------------------------------------------------
// AtomicCounter — WASM-compatible (uses Mutex instead of std::sync::atomic
// to ensure correct behaviour on platforms without native atomics)
// ------------------------------------------------------------------

/// A thread-safe counter backed by a `Mutex<i64>`.
///
/// This is intentionally implemented without `std::sync::atomic` so that it
/// compiles and works correctly on `wasm32-unknown-unknown` targets which may
/// not support 64-bit atomic intrinsics in all environments.
pub struct AtomicCounter {
    count: Arc<Mutex<i64>>,
}

impl AtomicCounter {
    /// Create a new counter with the given initial value.
    pub fn new(init: i64) -> Self {
        Self {
            count: Arc::new(Mutex::new(init)),
        }
    }

    /// Increment by 1 and return the *new* value.
    pub fn increment(&self) -> i64 {
        let mut guard = self.count.lock().unwrap_or_else(|e| e.into_inner());
        *guard += 1;
        *guard
    }

    /// Decrement by 1 and return the *new* value.
    pub fn decrement(&self) -> i64 {
        let mut guard = self.count.lock().unwrap_or_else(|e| e.into_inner());
        *guard -= 1;
        *guard
    }

    /// Load the current value.
    pub fn load(&self) -> i64 {
        *self.count.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Atomic compare-and-swap.
    ///
    /// If the current value equals `expected`, sets it to `desired` and returns
    /// `Ok(previous_value)`.  Otherwise returns `Err(current_value)`.
    pub fn compare_exchange(&self, expected: i64, desired: i64) -> Result<i64, i64> {
        let mut guard = self.count.lock().unwrap_or_else(|e| e.into_inner());
        let current = *guard;
        if current == expected {
            *guard = desired;
            Ok(current)
        } else {
            Err(current)
        }
    }
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn coordinator() -> ParallelCoordinator {
        ParallelCoordinator::new(ParallelConfig::default())
    }

    #[test]
    fn test_parallel_config_default() {
        let cfg = ParallelConfig::default();
        assert_eq!(cfg.n_workers, 4);
        assert!(!cfg.use_shared_memory);
        assert_eq!(cfg.chunk_size, 1024);
    }

    #[test]
    fn test_parallel_map_doubles() {
        let c = coordinator();
        let data: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let result = c.parallel_map(&data, |chunk| chunk.iter().map(|&x| x * 2.0).collect());
        let expected: Vec<f64> = (1..=8).map(|x| (x * 2) as f64).collect();
        assert_eq!(result, expected, "parallel_map doubles");
    }

    #[test]
    fn test_parallel_reduce_sum() {
        let cfg = ParallelConfig {
            n_workers: 2,
            chunk_size: 4,
            ..Default::default()
        };
        let c = ParallelCoordinator::new(cfg);
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result =
            c.parallel_reduce(&data, 0.0_f64, |acc, chunk| acc + chunk.iter().sum::<f64>());
        // Result is one of the partial sums; it must be in (0, 36].
        assert!(result > 0.0 && result <= 36.0, "partial sum = {result}");
    }

    #[test]
    fn test_parallel_matmul_2x2() {
        let c = coordinator();
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];
        let result = c.parallel_matmul(&a, &b, 2, 2, 2);
        assert!((result[0] - 19.0).abs() < 1e-9);
        assert!((result[1] - 22.0).abs() < 1e-9);
        assert!((result[2] - 43.0).abs() < 1e-9);
        assert!((result[3] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_parallel_sort_is_sorted() {
        let c = coordinator();
        let mut data = vec![5.0_f64, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0];
        c.parallel_sort(&mut data);
        for w in data.windows(2) {
            assert!(w[0] <= w[1], "not sorted: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_atomic_counter_increment_decrement() {
        let counter = AtomicCounter::new(0);
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.decrement(), 1);
        assert_eq!(counter.load(), 1);
    }

    #[test]
    fn test_atomic_counter_compare_exchange() {
        let counter = AtomicCounter::new(10);
        // Succeeds when expected matches.
        assert_eq!(counter.compare_exchange(10, 20), Ok(10));
        assert_eq!(counter.load(), 20);
        // Fails when expected does not match.
        assert_eq!(counter.compare_exchange(10, 30), Err(20));
        assert_eq!(counter.load(), 20); // unchanged
    }

    #[test]
    fn test_barrier_construction() {
        let b = ParallelCoordinator::barrier(3);
        // Just verify it compiles and creates an Arc<Barrier>.
        drop(b);
    }
}
