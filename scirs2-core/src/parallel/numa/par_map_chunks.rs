//! NUMA-locality parallel map over chunks.
//!
//! Provides `par_map_chunks`, a typed-result chunk map that builds a dedicated
//! rayon thread pool topology-matched to NUMA nodes on Linux and falls back
//! gracefully to plain rayon on Darwin/WASM or when NUMA detection returns `None`.

/// Environment variable that forces fully serial execution when set (any value).
///
/// Useful in tests, benchmarks, or constrained environments.
pub const SCIRS2_FORCE_SERIAL_ENV: &str = "SCIRS2_FORCE_SERIAL";

/// Apply `f` to each chunk of `input` (of size `chunk_size`) in parallel and
/// collect all output elements into a single `Vec<U>` preserving chunk order.
///
/// # Behaviour
///
/// 1. If the environment variable `SCIRS2_FORCE_SERIAL` is set (any value),
///    execution is fully serial — useful for deterministic debugging.
/// 2. If `input` is empty, an empty `Vec` is returned immediately.
/// 3. On Linux, a dedicated rayon `ThreadPool` is built whose thread count
///    matches the total online CPUs reported by `NumaTopology::detect()`.
///    Each worker thread is pinned to a NUMA node (by index modulo node count)
///    via `pthread_setaffinity_np`.  Affinity failures are silently swallowed —
///    the computation still proceeds, just without the locality hint.
/// 4. On non-Linux or when `NumaTopology::detect()` returns `None`, plain
///    `rayon::par_chunks` is used as a no-overhead fallback.
///
/// # Panics
///
/// Panics if `chunk_size == 0`.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "parallel")]
/// # {
/// use scirs2_core::par_map_chunks;
///
/// let input: Vec<i32> = (0..100).collect();
/// let doubled = par_map_chunks(&input, 10, |chunk| {
///     chunk.iter().map(|&x| x * 2).collect::<Vec<_>>()
/// });
/// assert_eq!(doubled.len(), 100);
/// # }
/// ```
pub fn par_map_chunks<T, U, F>(input: &[T], chunk_size: usize, f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&[T]) -> Vec<U> + Sync + Send,
{
    assert!(chunk_size > 0, "par_map_chunks: chunk_size must be > 0");

    // --- Serial fast-path via environment variable override ---
    if std::env::var(SCIRS2_FORCE_SERIAL_ENV).is_ok() {
        return input.chunks(chunk_size).flat_map(|c| f(c)).collect();
    }

    if input.is_empty() {
        return Vec::new();
    }

    // --- Try NUMA-aware thread pool (Linux only, requires `memory_efficient` + `libc`) ---
    #[cfg(all(target_os = "linux", feature = "memory_efficient", feature = "libc"))]
    {
        use crate::memory_efficient::numa_topology::NumaTopology;

        if let Some(topo) = NumaTopology::detect() {
            let node_count = topo.num_nodes();
            let total_cpus: usize = topo.nodes.iter().map(|n| n.cpu_list.len()).sum();
            // Clamp thread count: at least 1, at most total_cpus
            let num_threads = total_cpus.max(1);

            // Build per-NUMA-node CPU lists so the start_handler can pin without
            // referencing the topology struct (which isn't Send across the closure).
            // We collect a flat list: cpu_lists[thread_id % node_count] gives the
            // CPU mask for that thread.
            let per_node_cpus: Vec<Vec<usize>> =
                topo.nodes.iter().map(|n| n.cpu_list.clone()).collect();

            let pool_result = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .start_handler(move |tid| {
                    if node_count == 0 {
                        return;
                    }
                    let node_idx = tid % node_count;
                    if let Some(cpu_list) = per_node_cpus.get(node_idx) {
                        pin_thread_to_cpus(cpu_list);
                    }
                })
                .build();

            match pool_result {
                Ok(pool) => {
                    let chunks: Vec<&[T]> = input.chunks(chunk_size).collect();
                    return pool.install(|| {
                        use rayon::prelude::*;
                        chunks.into_par_iter().flat_map(|chunk| f(chunk)).collect()
                    });
                }
                Err(_) => {
                    // Fall through to the plain-rayon path below
                }
            }
        }
    }

    // --- Plain rayon fallback (non-Linux, no NUMA, or pool-build failure) ---
    {
        use rayon::prelude::*;
        input.par_chunks(chunk_size).flat_map(|c| f(c)).collect()
    }
}

/// Attempt to pin the calling thread to one of the listed CPU cores using
/// `pthread_setaffinity_np`.  Failures are silently ignored — the caller
/// continues executing without the locality hint.
///
/// This function is a no-op on non-Linux targets and is only compiled when the
/// `libc` feature is enabled (which is required for `pthread_setaffinity_np`).
#[cfg(all(target_os = "linux", feature = "libc"))]
fn pin_thread_to_cpus(cpu_list: &[usize]) {
    if cpu_list.is_empty() {
        return;
    }
    // Safety: we build the cpu_set_t correctly and immediately pass it to
    // pthread_setaffinity_np; no aliased pointer escapes this function.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpu_list {
            // cpu_set_t only holds up to CPU_SETSIZE CPUs; skip overflow.
            if cpu < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        let tid = libc::pthread_self();
        // Ignore the return value — affinity is a best-effort hint.
        let _ = libc::pthread_setaffinity_np(
            tid,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        );
    }
}
