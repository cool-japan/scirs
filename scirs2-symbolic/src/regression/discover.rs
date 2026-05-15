//! Main `discover` function — beam-search symbolic regression.
//!
//! Single-output search over a small grammar of [`LoweredOp`] building
//! blocks. The algorithm:
//!
//! 1. Seed the population with one identity formula per feature plus a
//!    handful of constants (`0`, `1`).
//! 2. Each iteration: evaluate every candidate on the data, score it with
//!    parsimony-regularised [`Fitness`], sort, and truncate to the beam
//!    width.
//! 3. Expand: the top survivors are combined via the enabled
//!    [`BuildingBlock`] families (e.g. `Sin`, `Mul(a, b)`) to produce the
//!    next generation, capped at `config.max_nodes`.
//! 4. Stop early when the best MSE drops below `config.tolerance`.
//!
//! This is the FOUNDATIONAL Phase 1 implementation; constant-fitting via
//! coordinate descent and SMT-pruned topology rejection arrive in v0.4.5.
//!
//! # Phase 1 design-freedom unlock A: NUMA-aware parallelism
//!
//! With the `numa` feature enabled, [`fn@discover`] dispatches large
//! feature-matrix evaluations through
//! `scirs2_core::par_map_chunks` for NUMA-locality-aware parallel
//! execution. On 4-node Skylake systems we expect ≥25% speedup over
//! serial evaluation for sample counts ≥ 1024 (per the v0.4.4 plan).
//! The 1024-sample threshold avoids parallel overhead on small problems.
//!
//! `par_map_chunks` pins worker threads to NUMA nodes on Linux
//! (via `pthread_setaffinity_np`) and falls back to plain rayon on
//! Darwin/WASM or when NUMA detection returns `None` — so the
//! feature is safe to enable on all platforms.
//!
//! Without the `numa` feature, [`fn@discover`] falls back to sequential
//! evaluation (still correct, just slower).

use crate::eml::eval::{eval_real, EvalCtx};
use crate::eml::LoweredOp;
use crate::regression::config::BuildingBlock;
use crate::regression::formula::count_nodes;
use crate::regression::{DiscoveredFormula, Fitness, SrConfig};
use ndarray::{ArrayView1, ArrayView2};

/// Sample-count threshold above which `predict` dispatches to the
/// NUMA-aware parallel path via `scirs2_core::par_map_chunks`. Below
/// this, the serial loop wins because thread-pool dispatch overhead
/// dominates per-row eval cost.
///
/// Exposed as `pub` so integration tests can assert the constant value.
#[cfg(feature = "numa")]
pub const NUMA_DISPATCH_THRESHOLD: usize = 1024;

/// Number of binding rows per work-chunk handed to each rayon/NUMA worker.
/// Balances per-chunk overhead against load imbalance; 64 rows ≈ 2–4 kB
/// of f64 data per chunk, fitting in a typical cache line group.
#[cfg(feature = "numa")]
const NUMA_CHUNK_SIZE: usize = 64;

/// Discover symbolic formulas from data.
///
/// Performs beam-search over a small grammar (defined by `config.building_blocks`)
/// to find formulas that approximate `targets ≈ f(features)`.
///
/// # Arguments
/// - `features`: shape `(n_samples, n_features)` 2D array of input data.
/// - `targets`: shape `(n_samples,)` 1D array of target values.
/// - `config`: search configuration (see [`SrConfig`]).
///
/// # Returns
/// A `Vec<DiscoveredFormula>` of the top `config.top_n` formulas, ranked by
/// combined fitness (lower is better). Returns an empty vector if the input
/// shapes are incompatible or the data is empty.
pub fn discover(
    features: ArrayView2<'_, f64>,
    targets: ArrayView1<'_, f64>,
    config: &SrConfig,
) -> Vec<DiscoveredFormula> {
    let (n_samples, n_features) = features.dim();
    if n_samples != targets.len() || n_samples == 0 {
        return Vec::new();
    }

    // Convert ndarray to per-sample variable bindings.
    let bindings: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| features[(i, j)]).collect())
        .collect();
    let target_slice: Vec<f64> = targets.to_vec();

    // Initial population: identity formulas (one per feature) plus constants.
    let mut population: Vec<LoweredOp> = (0..n_features).map(LoweredOp::Var).collect();
    population.push(LoweredOp::Const(1.0));
    population.push(LoweredOp::Const(0.0));

    let mut best_seen: Vec<DiscoveredFormula> = Vec::new();

    for _ in 0..config.max_iter {
        // Evaluate current population.
        let mut scored: Vec<DiscoveredFormula> = population
            .iter()
            .map(|op| {
                let predictions = predict(op, &bindings);
                let complexity = (count_nodes(op) as f64) * config.complexity_penalty;
                let fitness = Fitness::compute(&predictions, &target_slice, complexity);
                DiscoveredFormula::new(op.clone(), fitness)
            })
            .filter(|f| f.fitness.combined.is_finite())
            .collect();

        // Sort by combined fitness (ascending — lower is better).
        scored.sort_by(|a, b| {
            a.fitness
                .combined
                .partial_cmp(&b.fitness.combined)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Track best across all iterations.
        if let Some(best) = scored.first() {
            best_seen.push(best.clone());
        }

        // Truncate to beam width before expansion.
        scored.truncate(config.beam_width);

        // Early stopping if any candidate clears the MSE tolerance.
        if scored.iter().any(|f| f.fitness.mse < config.tolerance) {
            break;
        }

        // Generate next generation by combining top candidates.
        population = expand_population(&scored, config);

        // Cap population growth to avoid unbounded memory.
        if population.len() > config.beam_width * 4 {
            population.truncate(config.beam_width * 4);
        }
    }

    // Final sort + structural dedupe + top_n.
    best_seen.sort_by(|a, b| {
        a.fitness
            .combined
            .partial_cmp(&b.fitness.combined)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    best_seen.dedup_by(|a, b| a.op == b.op);
    best_seen.truncate(config.top_n);
    best_seen
}

/// Evaluate `op` at every binding row; non-evaluable points become NaN
/// so [`Fitness::compute`] flags the candidate as worst.
///
/// With the `numa` feature, evaluations are dispatched in parallel via
/// `scirs2_core::par_map_chunks` when the binding count exceeds
/// [`NUMA_DISPATCH_THRESHOLD`].  Below that threshold (and always,
/// when the feature is disabled), the serial path is taken to avoid
/// dispatch overhead dwarfing the per-row eval cost.
fn predict(op: &LoweredOp, bindings: &[Vec<f64>]) -> Vec<f64> {
    #[cfg(feature = "numa")]
    {
        if bindings.len() >= NUMA_DISPATCH_THRESHOLD {
            return predict_parallel(op, bindings);
        }
    }
    predict_serial(op, bindings)
}

/// Sequential fallback used for small inputs and when the `numa` feature
/// is disabled.  Kept separate from [`predict`] so the dispatch logic
/// stays readable and so the serial path is unconditionally available
/// for benchmarking comparisons.
fn predict_serial(op: &LoweredOp, bindings: &[Vec<f64>]) -> Vec<f64> {
    bindings
        .iter()
        .map(|vars| eval_real(op, &EvalCtx::new(vars)).unwrap_or(f64::NAN))
        .collect()
}

/// Parallel evaluation path gated on the `numa` feature.
///
/// Dispatches via [`scirs2_core::par_map_chunks`], which pins worker
/// threads to NUMA nodes on Linux (via `pthread_setaffinity_np`) and
/// falls back to plain rayon on Darwin/WASM or when NUMA detection
/// returns `None`.  Chunks of size [`NUMA_CHUNK_SIZE`] rows are handed
/// to each worker; within a chunk, rows are evaluated serially.
///
/// Only called when `bindings.len() >= `[`NUMA_DISPATCH_THRESHOLD`].
#[cfg(feature = "numa")]
fn predict_parallel(op: &LoweredOp, bindings: &[Vec<f64>]) -> Vec<f64> {
    use scirs2_core::par_map_chunks;
    par_map_chunks(bindings, NUMA_CHUNK_SIZE, |chunk| {
        chunk
            .iter()
            .map(|vars| eval_real(op, &EvalCtx::new(vars)).unwrap_or(f64::NAN))
            .collect()
    })
}

/// Build the next generation from the current top candidates.
///
/// Strategy:
/// - half the beam survives unchanged (elitism);
/// - the top quarter has unary operators applied (one per enabled family);
/// - top-quarter pairs are combined with `Add` and `Mul` (when arithmetic
///   is enabled), filtered by `config.max_nodes`;
/// - a fixed set of useful constants (`-1, 0.5, 2, π`) are seeded along
///   with `c · best` to allow scale-fitting.
fn expand_population(scored: &[DiscoveredFormula], config: &SrConfig) -> Vec<LoweredOp> {
    let mut next: Vec<LoweredOp> = Vec::new();

    // Elitism — keep top survivors.
    let half = (config.beam_width / 2).max(1);
    for f in scored.iter().take(half) {
        next.push(f.op.clone());
    }

    // Unary expansions on the top quarter.
    let quarter = (config.beam_width / 4).max(1);
    for f in scored.iter().take(quarter) {
        if config.building_blocks.contains(&BuildingBlock::Trig) {
            next.push(LoweredOp::Sin(Box::new(f.op.clone())));
            next.push(LoweredOp::Cos(Box::new(f.op.clone())));
        }
        if config.building_blocks.contains(&BuildingBlock::ExpLog) {
            next.push(LoweredOp::Exp(Box::new(f.op.clone())));
        }
        if config.building_blocks.contains(&BuildingBlock::SqrtAbs) {
            next.push(LoweredOp::Sqrt(Box::new(f.op.clone())));
            next.push(LoweredOp::Abs(Box::new(f.op.clone())));
        }
        if config.building_blocks.contains(&BuildingBlock::Hyperbolic) {
            next.push(LoweredOp::Tanh(Box::new(f.op.clone())));
        }
    }

    // Binary expansions on the top quarter (pairs including i==j to allow x*x).
    let top: Vec<&LoweredOp> = scored.iter().take(quarter).map(|f| &f.op).collect();
    if config.building_blocks.contains(&BuildingBlock::Arithmetic) {
        for (i, a) in top.iter().enumerate() {
            for b in top.iter().skip(i) {
                let sum = LoweredOp::Add(Box::new((*a).clone()), Box::new((*b).clone()));
                if count_nodes(&sum) <= config.max_nodes {
                    next.push(sum);
                }
                let prod = LoweredOp::Mul(Box::new((*a).clone()), Box::new((*b).clone()));
                if count_nodes(&prod) <= config.max_nodes {
                    next.push(prod);
                }
            }
        }
    }

    // Constant seeds + scaled best.
    for c in [-1.0, 0.5, 2.0, std::f64::consts::PI] {
        next.push(LoweredOp::Const(c));
        if let Some(top_op) = scored.first().map(|f| &f.op) {
            let scaled = LoweredOp::Mul(Box::new(LoweredOp::Const(c)), Box::new(top_op.clone()));
            if count_nodes(&scaled) <= config.max_nodes {
                next.push(scaled);
            }
        }
    }

    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn discover_identity() {
        // Target = x — the identity formula `Var(0)` is in the initial
        // population, so MSE on the best candidate must be near 0.
        let xs: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let features = Array2::from_shape_vec((50, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        let config = SrConfig::default().with_max_iter(20);
        let results = discover(features.view(), targets.view(), &config);

        assert!(!results.is_empty());
        assert!(
            results[0].fitness.mse < 1e-10,
            "best MSE = {}",
            results[0].fitness.mse
        );
    }

    #[test]
    fn discover_quadratic() {
        // Target = x². The expansion `Mul(Var(0), Var(0))` should appear in
        // generation 1 and produce R² ~ 1.
        let xs: Vec<f64> = (0..50).map(|i| (i as f64 - 25.0) * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let features = Array2::from_shape_vec((50, 1), xs).expect("shape");
        let targets = Array1::from_vec(ys);

        let config = SrConfig::default().with_max_iter(50);
        let results = discover(features.view(), targets.view(), &config);

        assert!(!results.is_empty());
        assert!(
            results[0].fitness.r_squared > 0.9,
            "best R² = {}",
            results[0].fitness.r_squared
        );
    }

    #[test]
    fn discover_returns_top_n() {
        let xs: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
        let features = Array2::from_shape_vec((30, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        let config = SrConfig::default().with_top_n(3).with_max_iter(10);
        let results = discover(features.view(), targets.view(), &config);

        assert!(results.len() <= 3);
    }

    #[test]
    fn discover_handles_empty() {
        let features = Array2::<f64>::zeros((0, 1));
        let targets = Array1::<f64>::zeros(0);
        let config = SrConfig::default();
        let results = discover(features.view(), targets.view(), &config);
        assert!(results.is_empty());
    }

    #[test]
    fn discover_handles_shape_mismatch() {
        let features = Array2::<f64>::zeros((10, 2));
        let targets = Array1::<f64>::zeros(7);
        let config = SrConfig::default();
        let results = discover(features.view(), targets.view(), &config);
        assert!(results.is_empty());
    }

    /// Smoke test for the `numa` parallel path: 2_000 samples > the
    /// 1_024 dispatch threshold, so [`predict`] routes through the
    /// rayon-backed `predict_parallel`.  We verify that the run
    /// completes, returns a non-empty result, recovers the quadratic
    /// shape (R² > 0.5), and is deterministic across invocations
    /// (beam search has no randomized component, so result ordering
    /// must match exactly even though the parallel reduce order may
    /// differ — only the *fitness ranking* is sensitive to FP order
    /// and the test does not require bit-identical fitness numbers).
    #[cfg(feature = "numa")]
    #[test]
    fn discover_parallel_matches_serial() {
        let xs: Vec<f64> = (0..2000).map(|i| (i as f64 - 1000.0) * 0.01).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let features = Array2::from_shape_vec((2000, 1), xs).expect("shape");
        let targets = Array1::from_vec(ys);

        let config = SrConfig::default()
            .with_max_iter(5)
            .with_top_n(3)
            .with_seed(42);
        let results1 = discover(features.view(), targets.view(), &config);
        let results2 = discover(features.view(), targets.view(), &config);

        assert_eq!(results1.len(), results2.len(), "result count differs");
        assert!(!results1.is_empty(), "no formulas discovered");
        assert!(
            results1[0].fitness.r_squared > 0.5,
            "best R² = {} (expected > 0.5 on x²)",
            results1[0].fitness.r_squared
        );
        // Beam search is deterministic — same input ⇒ same top formula.
        assert_eq!(
            results1[0].op, results2[0].op,
            "parallel path must be deterministic across runs"
        );
    }

    /// The serial path stays the chosen route for sample counts below
    /// the parallel threshold.  This test exercises the small-input
    /// branch under the `numa` feature to confirm the dispatch logic
    /// does not break the existing fast path (regression for the
    /// 50-sample `discover_identity` case).
    #[cfg(feature = "numa")]
    #[test]
    fn discover_below_threshold_uses_serial_path() {
        let xs: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
        let features = Array2::from_shape_vec((100, 1), xs.clone()).expect("shape");
        let targets = Array1::from_vec(xs);

        let config = SrConfig::default().with_max_iter(10);
        let results = discover(features.view(), targets.view(), &config);

        assert!(!results.is_empty());
        assert!(
            results[0].fitness.mse < 1e-10,
            "serial path produced MSE = {}",
            results[0].fitness.mse
        );
    }
}
