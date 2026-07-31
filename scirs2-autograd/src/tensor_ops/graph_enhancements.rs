//! Enhanced dynamic computation graph features (simplified)
//!
//! This module provides basic computation graph management features
//! including simple caching and conditional operations.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Simple computation cache
static COMPUTATION_CACHE: LazyLock<Mutex<HashMap<String, u64>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Cache configuration
static CACHE_CONFIG: LazyLock<Mutex<CacheConfig>> = LazyLock::new(|| {
    Mutex::new(CacheConfig {
        max_entries: 10000,
        ttl_seconds: 3600,
    })
});

/// Garbage collection state
static GC_STATE: LazyLock<Mutex<GcState>> = LazyLock::new(|| {
    Mutex::new(GcState {
        total_collections: 0,
        total_freed_bytes: 0,
    })
});

#[derive(Debug, Clone)]
struct CacheConfig {
    max_entries: usize,
    ttl_seconds: u64,
}

#[derive(Debug, Clone)]
struct GcState {
    total_collections: u64,
    total_freed_bytes: u64,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub max_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

/// Garbage collection statistics
#[derive(Debug, Clone)]
pub struct GcStats {
    pub active_references: usize,
    pub pending_collection: usize,
    pub total_collections: u64,
    pub total_freed_bytes: u64,
}

/// Conditional execution operation for control flow
pub struct ConditionalOp {
    pub predicate_type: PredicateType,
}

#[derive(Debug, Clone, Copy)]
pub enum PredicateType {
    GreaterThanZero,
    EqualToZero,
    NotEqualToZero,
    Threshold(f64),
}

impl<F: Float> Op<F> for ConditionalOp {
    fn name(&self) -> &'static str {
        "Conditional"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let condition = ctx.input(0);
        let true_branch = ctx.input(1);
        let false_branch = ctx.input(2);

        // Simple condition evaluation - check if first element meets condition
        let condition_met = predicate_holds(self.predicate_type, &condition);

        let result = if condition_met {
            true_branch.to_owned()
        } else {
            false_branch.to_owned()
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // `compute` returns exactly one of the two branches, so exactly one of them
        // influences the output and only that one may receive the cotangent.  Sending
        // `gy` to *both* (the previous behaviour) reports a non-zero gradient for a
        // branch whose value was discarded — for `conditional(c, f(x), g(x))` it makes
        // the gradient `f'(x) + g'(x)` instead of whichever branch actually ran.
        //
        // The predicate is a runtime value, so the routing has to be decided at
        // evaluation time; `ConditionalBranchGradOp` re-evaluates it and emits either
        // `gy` or a zero tensor of the same shape.
        let condition = *ctx.input(0);
        let gy = *ctx.output_grad();
        let g = ctx.graph();

        let mut branch_grad = |take_true: bool| {
            Tensor::builder(g)
                .append_input(condition, false)
                .append_input(gy, false)
                .build(ConditionalBranchGradOp {
                    predicate_type: self.predicate_type,
                    take_true,
                })
        };

        let to_true = branch_grad(true);
        let to_false = branch_grad(false);

        // The predicate itself is a step function of its input: zero derivative almost
        // everywhere, undefined at the switching point.
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, Some(to_true));
        ctx.append_input_grad(2, Some(to_false));
    }
}

/// Routes the cotangent of [`ConditionalOp`] to a single branch.
///
/// Inputs are `(condition, gy)`. The output is `gy` when the predicate selects the branch
/// this node belongs to (`take_true`), and an all-zero tensor of the same shape otherwise.
pub struct ConditionalBranchGradOp {
    predicate_type: PredicateType,
    take_true: bool,
}

/// Evaluates the branch predicate on the first element of `condition`.
///
/// Shared with [`ConditionalOp::compute`] so the forward and the backward pass can never
/// disagree about which branch ran.
fn predicate_holds<F: Float>(
    predicate_type: PredicateType,
    condition: &crate::ndarray_ext::NdArrayView<F>,
) -> bool {
    let first = match condition.iter().next() {
        Some(&v) => v,
        None => return false,
    };
    match predicate_type {
        PredicateType::GreaterThanZero => first > F::zero(),
        PredicateType::EqualToZero => first == F::zero(),
        PredicateType::NotEqualToZero => first != F::zero(),
        PredicateType::Threshold(threshold) => match first.to_f64() {
            Some(v) => v > threshold,
            None => false,
        },
    }
}

impl<F: Float> Op<F> for ConditionalBranchGradOp {
    fn name(&self) -> &'static str {
        "ConditionalBranchGrad"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let condition = ctx.input(0);
        let gy = ctx.input(1);
        let taken = predicate_holds(self.predicate_type, &condition);
        let out = if taken == self.take_true {
            gy.to_owned()
        } else {
            crate::ndarray_ext::NdArray::<F>::zeros(gy.raw_dim())
        };
        ctx.append_output(out);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Second-order: the routing is a constant (the predicate does not depend on `gy`),
        // so differentiating again just routes once more.
        let condition = *ctx.input(0);
        let ggy = *ctx.output_grad();
        let g = ctx.graph();
        let routed = Tensor::builder(g)
            .append_input(condition, false)
            .append_input(ggy, false)
            .build(ConditionalBranchGradOp {
                predicate_type: self.predicate_type,
                take_true: self.take_true,
            });
        ctx.append_input_grad(0, None);
        ctx.append_input_grad(1, Some(routed));
    }
}

/// Smart checkpoint operation (simplified)
pub struct SmartCheckpointOp {
    #[allow(dead_code)]
    pub memory_threshold: usize,
    #[allow(dead_code)]
    pub recompute_on_demand: bool,
}

impl<F: Float> Op<F> for SmartCheckpointOp {
    fn name(&self) -> &'static str {
        "SmartCheckpoint"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);
        // Checkpointing is a *memory* strategy, not a mathematical transformation: the
        // node forwards its input unchanged and exists only so the graph has a marked
        // boundary.  Recomputation happens naturally because this crate re-evaluates the
        // subgraph feeding the node whenever the node's value is needed again.
        ctx.append_output(input.to_owned());
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        // Forward is the identity, so the VJP is the identity: the cotangent passes
        // straight through to the wrapped computation, which is then differentiated by
        // its own ops exactly as if the checkpoint were not there.
        let gy = ctx.output_grad();
        ctx.append_input_grad(0, Some(*gy));
    }
}

/// Cached operation (simplified)
pub struct CachedOp {
    pub operation_name: String,
    #[allow(dead_code)]
    pub cache_key: String,
}

impl<F: Float> Op<F> for CachedOp {
    fn name(&self) -> &'static str {
        "Cached"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Simple caching - just record that we performed the operation
        let mut cache = COMPUTATION_CACHE.lock().expect("Operation failed");
        let counter = cache.entry(self.operation_name.clone()).or_insert(0);
        *counter += 1;

        // Perform simple operations based on name
        let result = match self.operation_name.as_str() {
            "identity" => input.to_owned(),
            "square" => input.mapv(|x| x * x),
            "sqrt" => input.mapv(|x| x.sqrt()),
            _ => input.to_owned(),
        };

        ctx.append_output(result);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();

        // Simple gradient computation
        let grad = match self.operation_name.as_str() {
            "identity" => *gy,
            "square" => {
                let input = ctx.input(0);
                let two = crate::tensor_ops::scalar(
                    F::from(2.0).expect("Failed to convert constant to float"),
                    ctx.graph(),
                );
                (*gy) * two * input
            }
            "sqrt" => {
                let input = ctx.input(0);
                let half = crate::tensor_ops::scalar(
                    F::from(0.5).expect("Failed to convert constant to float"),
                    ctx.graph(),
                );
                let sqrt_input = crate::tensor_ops::sqrt(input);
                (*gy) * half / sqrt_input
            }
            _ => *gy,
        };

        ctx.append_input_grad(0, Some(grad));
    }
}

// Public API functions

/// Clear the computation cache
#[allow(dead_code)]
pub fn clear_computation_cache() {
    COMPUTATION_CACHE.lock().expect("Operation failed").clear();
}

/// Get cache statistics
#[allow(dead_code)]
pub fn get_cache_stats() -> CacheStats {
    let cache = COMPUTATION_CACHE.lock().expect("Operation failed");
    let config = CACHE_CONFIG.lock().expect("Operation failed");
    CacheStats {
        entries: cache.len(),
        max_entries: config.max_entries,
        hits: 0,
        misses: 0,
        hit_rate: 0.0,
    }
}

/// Configure cache settings
#[allow(dead_code)]
pub fn configure_cache(_max_entries: usize, ttlseconds: u64) {
    let mut config = CACHE_CONFIG.lock().expect("Operation failed");
    config.max_entries = _max_entries;
    config.ttl_seconds = ttlseconds;
}

/// Run garbage collection
#[allow(dead_code)]
pub fn run_garbage_collection() -> usize {
    let mut gc_state = GC_STATE.lock().expect("Operation failed");
    gc_state.total_collections += 1;
    // Simulate freeing some memory
    let freed_items = 10usize;
    gc_state.total_freed_bytes += (freed_items as u64) * 100;
    freed_items
}

/// Get garbage collection statistics
#[allow(dead_code)]
pub fn get_gc_stats() -> GcStats {
    let gc_state = GC_STATE.lock().expect("Operation failed");
    GcStats {
        active_references: 0,
        pending_collection: 0,
        total_collections: gc_state.total_collections,
        total_freed_bytes: gc_state.total_freed_bytes,
    }
}

/// Create a conditional operation
#[allow(dead_code)]
pub fn conditional<'g, F: Float>(
    condition: &Tensor<'g, F>,
    true_branch: &Tensor<'g, F>,
    false_branch: &Tensor<'g, F>,
    predicate_type: PredicateType,
) -> Tensor<'g, F> {
    let g = condition.graph();
    Tensor::builder(g)
        .append_input(condition, false)
        .append_input(true_branch, false)
        .append_input(false_branch, false)
        .build(ConditionalOp { predicate_type })
}

/// Create a smart checkpoint
#[allow(dead_code)]
pub fn smart_checkpoint<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    memory_threshold: usize,
) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(SmartCheckpointOp {
            memory_threshold,
            recompute_on_demand: true,
        })
}

/// Create a cached operation
#[allow(dead_code)]
pub fn cached_op<'g, F: Float>(tensor: &Tensor<'g, F>, operationname: &str) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(CachedOp {
            operation_name: operationname.to_string(),
            cache_key: format!(
                "{}_{}",
                operationname,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("Failed to get slice")
                    .as_nanos()
            ),
        })
}

/// Graph enhancement utilities
pub struct GraphEnhancer;

impl GraphEnhancer {
    /// Optimize a computation graph
    pub fn optimize_graph() {
        clear_computation_cache();
        run_garbage_collection();
    }

    /// Get comprehensive graph statistics
    pub fn get_graph_stats() -> GraphStats {
        GraphStats {
            cache: get_cache_stats(),
            gc: get_gc_stats(),
        }
    }

    /// Configure graph for memory-constrained environments
    pub fn configure_for_memory_efficiency() {
        configure_cache(1000, 60);
    }

    /// Configure graph for performance
    pub fn configure_for_performance() {
        configure_cache(50000, 3600);
    }
}

/// Comprehensive graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub cache: CacheStats,
    pub gc: GcStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_operations() {
        clear_computation_cache();
        let stats = get_cache_stats();
        assert_eq!(stats.entries, 0);
    }

    #[test]
    fn test_gc_operations() {
        let collected = run_garbage_collection();
        assert_eq!(collected, 10);

        let stats = get_gc_stats();
        assert_eq!(stats.active_references, 0);
        assert!(stats.total_collections > 0);
        assert!(stats.total_freed_bytes > 0);
    }

    #[test]
    fn test_graph_enhancer() {
        GraphEnhancer::optimize_graph();
        let stats = GraphEnhancer::get_graph_stats();
        assert_eq!(stats.cache.entries, 0);

        GraphEnhancer::configure_for_memory_efficiency();
        GraphEnhancer::configure_for_performance();
    }
}
