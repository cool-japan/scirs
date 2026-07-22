//! NUMA-aware memory allocation strategies for high-performance computing.
//!
//! This module provides utilities for NUMA (Non-Uniform Memory Access) aware
//! memory management, which can significantly improve performance on multi-socket
//! systems by ensuring data is allocated close to the processor that will use it.
//!
//! On systems that do not expose NUMA topology (single-socket workstations, etc.)
//! the module transparently falls back to treating the whole machine as a single
//! NUMA node, so code using this module is portable without conditional
//! compilation.
//!
//! # Architecture
//!
//! ```text
//!   NumaTopology ──► [NumaNode {id, cpus, memory}]
//!         │
//!         ▼
//!   NumaAwarePool<T>
//!         │
//!         ├── per-node free-block cache   (Vec<Vec<T>>)
//!         └── NumaBuffer<T>  (owns Vec<T> + records node_id)
//! ```

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// NumaNode
// ---------------------------------------------------------------------------

/// Information about a single NUMA node.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Numeric NUMA node identifier (0-based).
    pub id: usize,
    /// Logical CPU IDs belonging to this node.
    pub cpu_ids: Vec<usize>,
    /// Total memory on this node in MiB (0 when unknown).
    pub memory_mb: usize,
    /// Free memory on this node in MiB (0 when unknown).
    pub free_memory_mb: usize,
}

// ---------------------------------------------------------------------------
// NumaTopology
// ---------------------------------------------------------------------------

/// Snapshot of the machine's NUMA topology.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// All NUMA nodes discovered.
    pub nodes: Vec<NumaNode>,
    /// Cached count of nodes (== `nodes.len()`).
    pub num_nodes: usize,
}

impl NumaTopology {
    /// Attempt to discover NUMA topology from the operating system.
    ///
    /// Discovery order (first success wins):
    ///
    /// 1. **sysfs** (Linux only) — parses
    ///    `/sys/devices/system/node/node*/cpulist` and `meminfo`.
    /// 2. **Single-node fallback** — treats the whole machine as one NUMA node.
    ///
    /// NUMA discovery via the `libnuma` C library was removed per the
    /// COOLJAPAN Pure Rust Policy: sysfs parsing is a complete, dependency-free
    /// functional replacement for the topology queries libnuma used to provide.
    pub fn discover() -> Self {
        #[cfg(target_os = "linux")]
        {
            if let Some(topo) = Self::discover_linux() {
                return topo;
            }
        }
        Self::single_node_fallback()
    }

    /// Build a single-node topology that covers all logical CPUs.
    fn single_node_fallback() -> Self {
        let cpu_count = num_cpus_count();
        let node = NumaNode {
            id: 0,
            cpu_ids: (0..cpu_count).collect(),
            memory_mb: 0,
            free_memory_mb: 0,
        };
        NumaTopology {
            num_nodes: 1,
            nodes: vec![node],
        }
    }

    /// Linux-specific topology discovery via sysfs.
    #[cfg(target_os = "linux")]
    fn discover_linux() -> Option<Self> {
        use std::fs;
        use std::path::Path;

        let node_base = Path::new("/sys/devices/system/node");
        if !node_base.exists() {
            return None;
        }

        let mut nodes: Vec<NumaNode> = Vec::new();

        let mut idx = 0usize;
        loop {
            let node_dir = node_base.join(format!("node{idx}"));
            if !node_dir.exists() {
                break;
            }

            // Parse CPU list (e.g. "0-3,8-11").
            let cpu_ids = fs::read_to_string(node_dir.join("cpulist"))
                .map(|s| parse_cpu_list(s.trim()))
                .unwrap_or_default();

            // Parse meminfo for MemTotal and MemFree.
            let (memory_mb, free_memory_mb) = fs::read_to_string(node_dir.join("meminfo"))
                .map(|s| parse_meminfo(&s))
                .unwrap_or((0, 0));

            // Skip memory-only nodes (no CPUs assigned), matching the
            // behavior of the removed libnuma-backed discovery path.
            if !cpu_ids.is_empty() {
                nodes.push(NumaNode {
                    id: idx,
                    cpu_ids,
                    memory_mb,
                    free_memory_mb,
                });
            }
            idx += 1;
        }

        if nodes.is_empty() {
            return None;
        }

        let num_nodes = nodes.len();
        Some(NumaTopology { nodes, num_nodes })
    }

    /// Return the NUMA node that owns `cpu_id`, or `None` if not found.
    pub fn node_for_cpu(&self, cpu_id: usize) -> Option<usize> {
        self.nodes
            .iter()
            .find(|n| n.cpu_ids.contains(&cpu_id))
            .map(|n| n.id)
    }

    /// Best-effort NUMA node for the currently running thread.
    ///
    /// Uses the CPU affinity mask on Linux; falls back to node 0 elsewhere.
    pub fn current_node(&self) -> usize {
        let cpu = current_cpu_id();
        self.node_for_cpu(cpu).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the number of logical CPUs without pulling in external crates.
fn num_cpus_count() -> usize {
    // std::thread::available_parallelism is stable since Rust 1.59.
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Best-effort current CPU id.  Returns 0 if unavailable.
fn current_cpu_id() -> usize {
    #[cfg(target_os = "linux")]
    {
        // sched_getcpu() is a cheap VDSO call on Linux.
        extern "C" {
            fn sched_getcpu() -> std::os::raw::c_int;
        }
        let cpu = unsafe { sched_getcpu() };
        if cpu >= 0 {
            return cpu as usize;
        }
    }
    0
}

/// Parse a Linux cpulist string like "0-3,8,10-11" into a Vec<usize>.
#[cfg(target_os = "linux")]
fn parse_cpu_list(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.trim().parse().unwrap_or(0);
            let hi: usize = hi.trim().parse().unwrap_or(0);
            for c in lo..=hi {
                cpus.push(c);
            }
        } else if let Ok(c) = part.parse::<usize>() {
            cpus.push(c);
        }
    }
    cpus
}

/// Parse Linux node meminfo and return (total_mb, free_mb).
#[cfg(target_os = "linux")]
fn parse_meminfo(s: &str) -> (usize, usize) {
    let mut total = 0usize;
    let mut free = 0usize;
    for line in s.lines() {
        if line.contains("MemTotal") {
            total = extract_kb(line) / 1024;
        } else if line.contains("MemFree") {
            free = extract_kb(line) / 1024;
        }
    }
    (total, free)
}

#[cfg(target_os = "linux")]
fn extract_kb(line: &str) -> usize {
    // Lines look like: "Node 0 MemTotal:  16777216 kB"
    line.split_whitespace()
        .find_map(|w| w.parse::<usize>().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// NumaBuffer
// ---------------------------------------------------------------------------

/// A heap-allocated buffer tagged with the NUMA node it logically belongs to.
///
/// Allocation itself is performed by the standard allocator; the `node_id` tag
/// is advisory and can be used by higher-level algorithms to prefer data
/// movement within a node.
pub struct NumaBuffer<T> {
    data: Vec<T>,
    node_id: usize,
}

impl<T: Default + Clone> NumaBuffer<T> {
    /// Allocate a zero-initialised buffer of `size` elements tagged to `node_id`.
    pub fn new(size: usize, node_id: usize) -> Self {
        NumaBuffer {
            data: vec![T::default(); size],
            node_id,
        }
    }

    /// Shared slice view.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable slice view.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// The NUMA node this buffer was tagged with.
    pub fn node_id(&self) -> usize {
        self.node_id
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` iff the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// NumaAwarePool
// ---------------------------------------------------------------------------

/// A per-NUMA-node pool of reusable fixed-size blocks.
///
/// Blocks are allocated on demand and returned to the per-node cache when
/// released.  This avoids repeated heap allocations in hot loops while keeping
/// locality information.
///
/// # Type parameter
///
/// `T` must implement `Default + Clone + Send`.  The `Send` bound is required
/// because blocks can be moved between threads (caller's responsibility to
/// only hand a block to a thread that matches its node affinity).
pub struct NumaAwarePool<T: Default + Clone + Send> {
    /// `per_node_pools[node_id]` is a stack of cached free blocks.
    per_node_pools: Vec<Vec<Vec<T>>>,
    block_size: usize,
    topology: NumaTopology,
}

impl<T: Default + Clone + Send> NumaAwarePool<T> {
    /// Create a new pool with the specified `block_size`.
    ///
    /// The topology is discovered automatically.
    pub fn new(block_size: usize) -> Self {
        let topology = NumaTopology::discover();
        let num_nodes = topology.num_nodes;
        NumaAwarePool {
            per_node_pools: vec![Vec::new(); num_nodes],
            block_size,
            topology,
        }
    }

    /// Allocate a block.
    ///
    /// If `node_id` is `Some(n)` and `n` is valid, a cached block from that
    /// node is returned (or a fresh one allocated and tagged).  `None` selects
    /// the current thread's preferred node.
    pub fn allocate(&mut self, node_id: Option<usize>) -> Vec<T> {
        let node = self.resolve_node(node_id);
        if let Some(block) = self.per_node_pools[node].pop() {
            return block;
        }
        vec![T::default(); self.block_size]
    }

    /// Return a block to the pool.
    ///
    /// `node_id` semantics are the same as for [`allocate`](Self::allocate).
    /// If the block's length differs from `block_size` it is discarded.
    pub fn deallocate(&mut self, block: Vec<T>, node_id: Option<usize>) {
        if block.len() != self.block_size {
            return; // Discard non-conforming blocks silently.
        }
        let node = self.resolve_node(node_id);
        self.per_node_pools[node].push(block);
    }

    /// Return `(node_id, cached_block_count)` for every node.
    pub fn stats(&self) -> Vec<(usize, usize)> {
        self.per_node_pools
            .iter()
            .enumerate()
            .map(|(i, pool)| (i, pool.len()))
            .collect()
    }

    fn resolve_node(&self, hint: Option<usize>) -> usize {
        let n = match hint {
            Some(id) => id,
            None => self.topology.current_node(),
        };
        n.min(self.topology.num_nodes.saturating_sub(1))
    }
}

// ---------------------------------------------------------------------------
// Validate topology helper (public utility)
// ---------------------------------------------------------------------------

/// Validate that a node_id is within the bounds of the topology.
pub fn validate_node_id(topology: &NumaTopology, node_id: usize) -> CoreResult<()> {
    if node_id < topology.num_nodes {
        Ok(())
    } else {
        Err(CoreError::InvalidArgument(ErrorContext::new(format!(
            "NUMA node_id {node_id} is out of range (topology has {} nodes)",
            topology.num_nodes
        ))))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_discover_returns_at_least_one_node() {
        let topo = NumaTopology::discover();
        assert!(topo.num_nodes >= 1);
        assert_eq!(topo.nodes.len(), topo.num_nodes);
        for node in &topo.nodes {
            assert!(!node.cpu_ids.is_empty());
        }
    }

    #[test]
    fn test_current_node_within_bounds() {
        let topo = NumaTopology::discover();
        let cur = topo.current_node();
        assert!(cur < topo.num_nodes);
    }

    #[test]
    fn test_numa_buffer_basic() {
        let mut buf: NumaBuffer<f64> = NumaBuffer::new(1024, 0);
        assert_eq!(buf.len(), 1024);
        assert!(!buf.is_empty());
        assert_eq!(buf.node_id(), 0);

        // All elements default-initialised to 0.0.
        assert!(buf.as_slice().iter().all(|&v| v == 0.0));

        // Mutate via slice.
        buf.as_mut_slice()[0] = 3.15;
        assert_eq!(buf.as_slice()[0], 3.15);
    }

    #[test]
    fn test_numa_buffer_empty() {
        let buf: NumaBuffer<u8> = NumaBuffer::new(0, 0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_pool_allocate_deallocate() {
        let mut pool: NumaAwarePool<u64> = NumaAwarePool::new(64);

        // Allocate from node 0.
        let block = pool.allocate(Some(0));
        assert_eq!(block.len(), 64);
        assert!(block.iter().all(|&v| v == 0));

        // Return to pool.
        pool.deallocate(block, Some(0));

        let stats = pool.stats();
        assert_eq!(stats[0].1, 1); // One cached block at node 0.

        // Re-allocate reuses the cached block.
        let _block2 = pool.allocate(Some(0));
        let stats2 = pool.stats();
        assert_eq!(stats2[0].1, 0); // Cache should be empty now.
    }

    #[test]
    fn test_pool_wrong_size_discarded() {
        let mut pool: NumaAwarePool<u32> = NumaAwarePool::new(32);
        // Deallocate a block with the wrong size; it should be discarded.
        pool.deallocate(vec![0u32; 16], Some(0));
        let stats = pool.stats();
        assert_eq!(stats[0].1, 0);
    }

    #[test]
    fn test_pool_current_node_allocation() {
        let mut pool: NumaAwarePool<i32> = NumaAwarePool::new(8);
        // Allocate without specifying node; should use current node.
        let block = pool.allocate(None);
        assert_eq!(block.len(), 8);
        pool.deallocate(block, None);
    }

    #[test]
    fn test_validate_node_id() {
        let topo = NumaTopology::discover();
        assert!(validate_node_id(&topo, 0).is_ok());
        assert!(validate_node_id(&topo, topo.num_nodes).is_err());
    }

    #[test]
    fn test_node_for_cpu() {
        let topo = NumaTopology::discover();
        // CPU 0 must belong to some node.
        let node = topo.node_for_cpu(0);
        assert!(node.is_some());
        // A very large CPU id should not be found.
        assert!(topo.node_for_cpu(usize::MAX).is_none());
    }
}
