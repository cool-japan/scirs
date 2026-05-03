//! Performance profiling utilities for neural networks
//!
//! This module provides tools for measuring and analysing the computational
//! cost of neural network layers and operations:
//!
//! - [`PerformanceProfiler`] – profile individual layers
//! - [`LayerStats`] – per-layer timing and memory statistics
//! - [`FLOPsCounter`] – estimate FLOPs for common layer types
//! - [`ProfilingReport`] – aggregated results from a full profiling run

use crate::error::Result;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// LayerStats
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer performance statistics collected by [`PerformanceProfiler`].
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Layer name
    pub layer_name: String,
    /// Forward pass wall-clock time
    pub forward_ms: f64,
    /// Backward pass wall-clock time (0 if not measured)
    pub backward_ms: f64,
    /// Approximate memory allocated for activations in bytes
    pub memory_bytes: usize,
    /// Estimated floating-point operations (multiplications + additions)
    pub flops: u64,
    /// Number of trainable parameters
    pub param_count: usize,
    /// Number of forward profiling invocations recorded
    pub num_forward_runs: usize,
    /// Number of backward profiling invocations recorded
    pub num_backward_runs: usize,
}

impl LayerStats {
    /// Create a zero-initialised stat record for the given layer.
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            layer_name: layer_name.into(),
            forward_ms: 0.0,
            backward_ms: 0.0,
            memory_bytes: 0,
            flops: 0,
            param_count: 0,
            num_forward_runs: 0,
            num_backward_runs: 0,
        }
    }

    /// Returns average forward time in milliseconds.
    pub fn avg_forward_ms(&self) -> f64 {
        if self.num_forward_runs == 0 {
            return 0.0;
        }
        self.forward_ms / self.num_forward_runs as f64
    }

    /// Returns average backward time in milliseconds.
    pub fn avg_backward_ms(&self) -> f64 {
        if self.num_backward_runs == 0 {
            return 0.0;
        }
        self.backward_ms / self.num_backward_runs as f64
    }

    /// Returns memory in megabytes.
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns FLOPs in giga-FLOP units.
    pub fn gflops(&self) -> f64 {
        self.flops as f64 / 1e9
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PerformanceProfiler
// ─────────────────────────────────────────────────────────────────────────────

/// Profiles neural network layer performance by timing closures.
///
/// # Examples
/// ```
/// use scirs2_neural::performance::PerformanceProfiler;
///
/// let mut profiler = PerformanceProfiler::new();
///
/// let stats = profiler.profile_layer("dense_1", || {
///     // Simulate a forward pass
///     let sum: f64 = (0..1000).map(|i| i as f64).sum();
///     (sum, 1000 * 4) // (result, bytes_allocated)
/// }, None)
/// .expect("profile ok");
///
/// assert!(stats.forward_ms >= 0.0);
/// assert_eq!(stats.memory_bytes, 1000 * 4);
/// ```
pub struct PerformanceProfiler {
    layer_stats: HashMap<String, LayerStats>,
    /// Number of warm-up runs before measurement
    pub warmup_runs: usize,
    /// Number of timed runs to average
    pub timed_runs: usize,
}

impl PerformanceProfiler {
    /// Create a profiler with default settings (1 warmup, 3 timed runs).
    pub fn new() -> Self {
        Self {
            layer_stats: HashMap::new(),
            warmup_runs: 1,
            timed_runs: 3,
        }
    }

    /// Profile a forward pass closure for the named layer.
    ///
    /// The closure must return `(value, memory_bytes)` where `memory_bytes` is
    /// the approximate number of bytes allocated for the output activations.
    ///
    /// # Arguments
    /// * `layer_name` – identifier for the layer
    /// * `forward_fn` – the forward pass to time; returns `(T, usize)` where the
    ///   second element is the estimated activation memory in bytes.
    /// * `flops` – optional estimated FLOPs; computed by `FLOPsCounter` methods if
    ///   known ahead of time.
    pub fn profile_layer<T, F>(
        &mut self,
        layer_name: &str,
        forward_fn: F,
        flops: Option<u64>,
    ) -> Result<LayerStats>
    where
        F: Fn() -> (T, usize),
    {
        // Warm-up runs (results discarded)
        for _ in 0..self.warmup_runs {
            forward_fn();
        }

        // Timed runs
        let mut total_ns: u128 = 0;
        let mut memory_bytes = 0usize;
        for _ in 0..self.timed_runs.max(1) {
            let start = Instant::now();
            let (_, mem) = forward_fn();
            let elapsed = start.elapsed();
            total_ns += elapsed.as_nanos();
            memory_bytes = mem; // last measurement wins
        }
        let n_runs = self.timed_runs.max(1) as u128;
        let avg_ns = total_ns / n_runs;
        let forward_ms = Duration::from_nanos(avg_ns as u64).as_secs_f64() * 1000.0;

        let stats = LayerStats {
            layer_name: layer_name.to_string(),
            forward_ms,
            backward_ms: 0.0,
            memory_bytes,
            flops: flops.unwrap_or(0),
            param_count: 0,
            num_forward_runs: self.timed_runs.max(1),
            num_backward_runs: 0,
        };

        self.layer_stats
            .insert(layer_name.to_string(), stats.clone());
        Ok(stats)
    }

    /// Profile both a forward and backward pass closure for the named layer.
    ///
    /// `forward_fn` returns `(T, usize)` (value + memory bytes).
    /// `backward_fn` takes the forward output and computes gradients.
    pub fn profile_layer_with_backward<T, F, B>(
        &mut self,
        layer_name: &str,
        forward_fn: F,
        backward_fn: B,
        flops: Option<u64>,
    ) -> Result<LayerStats>
    where
        F: Fn() -> (T, usize),
        B: Fn(T),
    {
        // Warm-up
        for _ in 0..self.warmup_runs {
            let (val, _) = forward_fn();
            backward_fn(val);
        }

        // Timed runs
        let mut fwd_ns: u128 = 0;
        let mut bwd_ns: u128 = 0;
        let mut memory_bytes = 0usize;
        for _ in 0..self.timed_runs.max(1) {
            let fwd_start = Instant::now();
            let (val, mem) = forward_fn();
            fwd_ns += fwd_start.elapsed().as_nanos();
            memory_bytes = mem;

            let bwd_start = Instant::now();
            backward_fn(val);
            bwd_ns += bwd_start.elapsed().as_nanos();
        }
        let n = self.timed_runs.max(1) as u128;

        let forward_ms = Duration::from_nanos((fwd_ns / n) as u64).as_secs_f64() * 1000.0;
        let backward_ms = Duration::from_nanos((bwd_ns / n) as u64).as_secs_f64() * 1000.0;

        let stats = LayerStats {
            layer_name: layer_name.to_string(),
            forward_ms,
            backward_ms,
            memory_bytes,
            flops: flops.unwrap_or(0),
            param_count: 0,
            num_forward_runs: self.timed_runs.max(1),
            num_backward_runs: self.timed_runs.max(1),
        };

        self.layer_stats
            .insert(layer_name.to_string(), stats.clone());
        Ok(stats)
    }

    /// Return stats for a previously profiled layer, if available.
    pub fn get_stats(&self, layer_name: &str) -> Option<&LayerStats> {
        self.layer_stats.get(layer_name)
    }

    /// Return all recorded layer stats, sorted by average forward time (descending).
    pub fn all_stats_sorted(&self) -> Vec<&LayerStats> {
        let mut v: Vec<&LayerStats> = self.layer_stats.values().collect();
        v.sort_by(|a, b| {
            b.avg_forward_ms()
                .partial_cmp(&a.avg_forward_ms())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Reset all recorded statistics.
    pub fn reset(&mut self) {
        self.layer_stats.clear();
    }

    /// Build a [`ProfilingReport`] from all recorded stats.
    pub fn report(&self) -> ProfilingReport {
        let stats: Vec<LayerStats> = self.layer_stats.values().cloned().collect();
        ProfilingReport::from_stats(stats)
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FLOPsCounter
// ─────────────────────────────────────────────────────────────────────────────

/// Estimates FLOPs (floating-point operations = MACs × 2) for common layer types.
///
/// Counts **multiply-accumulate operations × 2** (one multiply + one addition).
/// These are coarse estimates and match the convention used in most ML papers.
pub struct FLOPsCounter;

impl FLOPsCounter {
    /// FLOPs for a dense (fully-connected) layer.
    ///
    /// `2 × batch × in_features × out_features`
    pub fn dense(batch: usize, in_features: usize, out_features: usize) -> u64 {
        2 * batch as u64 * in_features as u64 * out_features as u64
    }

    /// FLOPs for a standard 2-D convolution.
    ///
    /// `2 × batch × out_ch × H_out × W_out × in_ch × kH × kW`
    pub fn conv2d(
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        out_height: usize,
        out_width: usize,
    ) -> u64 {
        2 * batch as u64
            * out_channels as u64
            * out_height as u64
            * out_width as u64
            * in_channels as u64
            * kernel_h as u64
            * kernel_w as u64
    }

    /// FLOPs for a depthwise 2-D convolution.
    ///
    /// `2 × batch × channels × H_out × W_out × kH × kW`
    pub fn depthwise_conv2d(
        batch: usize,
        channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        out_height: usize,
        out_width: usize,
    ) -> u64 {
        2 * batch as u64
            * channels as u64
            * out_height as u64
            * out_width as u64
            * kernel_h as u64
            * kernel_w as u64
    }

    /// FLOPs for a pointwise (1×1) convolution.
    pub fn pointwise_conv2d(
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    ) -> u64 {
        2 * batch as u64 * in_channels as u64 * out_channels as u64 * height as u64 * width as u64
    }

    /// FLOPs for a scaled dot-product attention block.
    ///
    /// Counts Q·K^T (`2 × L × L × d_k`) + softmax (approx `4 × L²`)
    /// + attention·V (`2 × L² × d_v`).
    ///
    /// `batch × heads × (2 × seq_len² × d_key + 4 × seq_len² + 2 × seq_len² × d_val)`
    pub fn attention(
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        d_key: usize,
        d_val: usize,
    ) -> u64 {
        let l = seq_len as u64;
        let dk = d_key as u64;
        let dv = d_val as u64;
        let per_head = 2 * l * l * dk  // QK^T
            + 4 * l * l     // softmax approx
            + 2 * l * l * dv; // attn·V
        batch as u64 * num_heads as u64 * per_head
    }

    /// FLOPs for a batch normalisation layer (2 ops per element: normalise + scale/shift).
    pub fn batch_norm(batch: usize, channels: usize, height: usize, width: usize) -> u64 {
        2 * batch as u64 * channels as u64 * height as u64 * width as u64
    }

    /// FLOPs for layer normalisation (same formula as batch norm but over feature dim).
    pub fn layer_norm(batch: usize, seq_len: usize, hidden_dim: usize) -> u64 {
        2 * batch as u64 * seq_len as u64 * hidden_dim as u64
    }

    /// FLOPs for an embedding lookup (essentially free – just an index op, returns 0).
    pub fn embedding(_batch: usize, _seq_len: usize, _embedding_dim: usize) -> u64 {
        0
    }

    /// FLOPs for an LSTM cell (approximate).
    ///
    /// 4 gates × (2 × hidden × batch + 2 × input × batch) + element-wise ops
    pub fn lstm_cell(batch: usize, input_size: usize, hidden_size: usize, seq_len: usize) -> u64 {
        let per_step = 8 * batch as u64 * hidden_size as u64
            + 8 * batch as u64 * input_size as u64
            + 12 * batch as u64 * hidden_size as u64; // element-wise
        per_step * seq_len as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProfilingReport
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated profiling report for all measured layers.
#[derive(Debug, Clone)]
pub struct ProfilingReport {
    /// Stats per layer
    pub layers: Vec<LayerStats>,
    /// Total forward time across all layers (ms)
    pub total_forward_ms: f64,
    /// Total backward time across all layers (ms)
    pub total_backward_ms: f64,
    /// Total memory across all layers (bytes)
    pub total_memory_bytes: usize,
    /// Total estimated FLOPs
    pub total_flops: u64,
    /// Name of the slowest layer
    pub bottleneck_layer: Option<String>,
}

impl ProfilingReport {
    /// Build a report from a vector of layer stats.
    pub fn from_stats(layers: Vec<LayerStats>) -> Self {
        let total_forward_ms: f64 = layers.iter().map(|s| s.avg_forward_ms()).sum();
        let total_backward_ms: f64 = layers.iter().map(|s| s.avg_backward_ms()).sum();
        let total_memory_bytes: usize = layers.iter().map(|s| s.memory_bytes).sum();
        let total_flops: u64 = layers.iter().map(|s| s.flops).sum();

        let bottleneck_layer = layers
            .iter()
            .max_by(|a, b| {
                a.avg_forward_ms()
                    .partial_cmp(&b.avg_forward_ms())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.layer_name.clone());

        Self {
            layers,
            total_forward_ms,
            total_backward_ms,
            total_memory_bytes,
            total_flops,
            bottleneck_layer,
        }
    }

    /// Returns total memory in megabytes.
    pub fn total_memory_mb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Returns total FLOPs in Giga-FLOP units.
    pub fn total_gflops(&self) -> f64 {
        self.total_flops as f64 / 1e9
    }

    /// Returns layers sorted by forward time (slowest first).
    pub fn layers_by_forward_time(&self) -> Vec<&LayerStats> {
        let mut v: Vec<&LayerStats> = self.layers.iter().collect();
        v.sort_by(|a, b| {
            b.avg_forward_ms()
                .partial_cmp(&a.avg_forward_ms())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_layer_basic() {
        let mut profiler = PerformanceProfiler::new();
        let stats = profiler
            .profile_layer(
                "dense_1",
                || {
                    let _: f64 = (0..1000).map(|i| i as f64).sum();
                    ((), 1000 * 8)
                },
                Some(2_000),
            )
            .expect("ok");
        assert_eq!(stats.layer_name, "dense_1");
        assert!(stats.forward_ms >= 0.0);
        assert_eq!(stats.memory_bytes, 1000 * 8);
        assert_eq!(stats.flops, 2_000);
    }

    #[test]
    fn test_profile_layer_with_backward() {
        let mut profiler = PerformanceProfiler::new();
        let stats = profiler
            .profile_layer_with_backward(
                "layer_a",
                || (vec![1.0_f64; 64], 64 * 8),
                |v: Vec<f64>| {
                    let _: f64 = v.iter().sum();
                },
                None,
            )
            .expect("ok");
        assert!(stats.backward_ms >= 0.0);
        assert_eq!(stats.num_backward_runs, profiler.timed_runs.max(1));
    }

    #[test]
    fn test_profiler_get_stats() {
        let mut profiler = PerformanceProfiler::new();
        profiler
            .profile_layer("conv1", || ((), 512 * 64), None)
            .expect("ok");
        assert!(profiler.get_stats("conv1").is_some());
        assert!(profiler.get_stats("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = PerformanceProfiler::new();
        profiler.profile_layer("l1", || ((), 0), None).expect("ok");
        assert!(!profiler.layer_stats.is_empty());
        profiler.reset();
        assert!(profiler.layer_stats.is_empty());
    }

    #[test]
    fn test_profiler_report() {
        let mut profiler = PerformanceProfiler::new();
        profiler
            .profile_layer("l1", || ((), 100), Some(1000))
            .expect("ok");
        profiler
            .profile_layer("l2", || ((), 200), Some(2000))
            .expect("ok");
        let report = profiler.report();
        assert_eq!(report.layers.len(), 2);
        assert_eq!(report.total_flops, 3000);
        assert_eq!(report.total_memory_bytes, 300);
    }

    #[test]
    fn test_all_stats_sorted() {
        let mut profiler = PerformanceProfiler::new();
        profiler.warmup_runs = 0;
        profiler.timed_runs = 1;
        profiler
            .profile_layer("fast", || ((), 0), None)
            .expect("ok");
        // Make slow layer actually compute something
        profiler
            .profile_layer(
                "slow",
                || {
                    let _: u64 = (0u64..100_000).sum();
                    ((), 0)
                },
                None,
            )
            .expect("ok");
        let sorted = profiler.all_stats_sorted();
        // Just check both layers are present
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn test_flops_counter_dense() {
        let flops = FLOPsCounter::dense(1, 784, 512);
        assert_eq!(flops, 2 * 784 * 512); // batch=1
    }

    #[test]
    fn test_flops_counter_conv2d() {
        let flops = FLOPsCounter::conv2d(1, 3, 64, 3, 3, 224, 224);
        let expected = 2u64 * 64 * 224 * 224 * 3 * 3 * 3; // batch=1
        assert_eq!(flops, expected);
    }

    #[test]
    fn test_flops_counter_depthwise() {
        let flops = FLOPsCounter::depthwise_conv2d(1, 32, 3, 3, 112, 112);
        let expected = 2u64 * 32 * 112 * 112 * 3 * 3; // batch=1
        assert_eq!(flops, expected);
    }

    #[test]
    fn test_flops_counter_attention() {
        let flops = FLOPsCounter::attention(1, 8, 128, 64, 64);
        assert!(flops > 0);
    }

    #[test]
    fn test_flops_counter_batch_norm() {
        let flops = FLOPsCounter::batch_norm(4, 64, 56, 56);
        assert_eq!(flops, 2 * 4 * 64 * 56 * 56);
    }

    #[test]
    fn test_flops_counter_embedding_is_zero() {
        assert_eq!(FLOPsCounter::embedding(2, 512, 768), 0);
    }

    #[test]
    fn test_layer_stats_avg() {
        let mut s = LayerStats::new("test");
        s.forward_ms = 6.0;
        s.num_forward_runs = 3;
        assert!((s.avg_forward_ms() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_layer_stats_memory_mb() {
        let mut s = LayerStats::new("x");
        s.memory_bytes = 1024 * 1024;
        assert!((s.memory_mb() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_profiling_report_bottleneck() {
        let mut stats = vec![
            LayerStats {
                layer_name: "fast".to_string(),
                forward_ms: 1.0,
                num_forward_runs: 1,
                ..LayerStats::new("fast")
            },
            LayerStats {
                layer_name: "slow".to_string(),
                forward_ms: 10.0,
                num_forward_runs: 1,
                ..LayerStats::new("slow")
            },
        ];
        let report = ProfilingReport::from_stats(std::mem::take(&mut stats));
        assert_eq!(report.bottleneck_layer.as_deref(), Some("slow"));
    }
}
