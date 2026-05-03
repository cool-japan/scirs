//! Typed data transformation pipeline
//!
//! Provides a strongly-typed, composable pipeline for data transformations.
//! Transforms are chained via `Pipeline<I, O>` which tracks the input and
//! output types at compile time.
//!
//! # Example
//! ```rust,ignore
//! use scirs2_io::pipeline::typed_transforms::*;
//!
//! let pipeline = TypedPipeline::new()
//!     .then(MapTransform::new(|x: i32| x * 2))
//!     .then(FilterTransform::new(|x: &i32| *x > 4));
//!
//! let result = pipeline.apply(3).unwrap();
//! assert_eq!(result, Some(6));
//! ```

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Core trait
// ─────────────────────────────────────────────────────────────────────────────

/// A typed transformation from `Input` to `Output`.
pub trait Transform: Send + Sync {
    /// Input element type
    type Input;
    /// Output element type
    type Output;

    /// Transform a single item.
    fn transform(&self, input: Self::Input) -> Result<Self::Output>;

    /// Human-readable name used for diagnostics.
    fn name(&self) -> &str {
        "anonymous"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type-erased internal representation
// ─────────────────────────────────────────────────────────────────────────────

/// Internal type-erased step stored inside a `Pipeline`.
trait PipelineStep<I, O>: Send + Sync {
    fn apply(&self, input: I) -> Result<O>;
    fn name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A composable, type-safe chain of [`Transform`]s.
///
/// `I` is the pipeline's overall input type; `O` is the overall output type.
/// Intermediate types are erased behind `Arc<dyn PipelineStep<…>>`.
pub struct TypedPipeline<I, O> {
    step: Arc<dyn PipelineStep<I, O>>,
}

// identity pipeline (I == O)
struct IdentityStep<T>(PhantomData<T>);

impl<T: Send + Sync> PipelineStep<T, T> for IdentityStep<T> {
    fn apply(&self, input: T) -> Result<T> {
        Ok(input)
    }
    fn name(&self) -> &str {
        "identity"
    }
}

impl<I: Send + Sync + 'static> TypedPipeline<I, I> {
    /// Create an empty (identity) pipeline.
    pub fn new() -> Self {
        Self {
            step: Arc::new(IdentityStep(PhantomData)),
        }
    }
}

impl<I: Send + Sync + 'static> Default for TypedPipeline<I, I> {
    fn default() -> Self {
        Self::new()
    }
}

/// Composed step: run `first` then `second`.
struct ComposedStep<A, B, C> {
    first: Arc<dyn PipelineStep<A, B>>,
    second: Arc<dyn PipelineStep<B, C>>,
}

impl<A, B, C> PipelineStep<A, C> for ComposedStep<A, B, C>
where
    A: Send + Sync + 'static,
    B: Send + Sync + 'static,
    C: Send + Sync + 'static,
{
    fn apply(&self, input: A) -> Result<C> {
        let intermediate = self.first.apply(input)?;
        self.second.apply(intermediate)
    }
    fn name(&self) -> &str {
        "composed"
    }
}

/// Adapter that wraps a [`Transform`] into a [`PipelineStep`].
struct TransformStep<T: Transform> {
    transform: T,
}

impl<T: Transform> PipelineStep<T::Input, T::Output> for TransformStep<T>
where
    T::Input: Send + Sync + 'static,
    T::Output: Send + Sync + 'static,
{
    fn apply(&self, input: T::Input) -> Result<T::Output> {
        self.transform.transform(input)
    }
    fn name(&self) -> &str {
        self.transform.name()
    }
}

impl<I, O> TypedPipeline<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Append a new transform, extending the output type to `T::Output`.
    pub fn then<T>(self, transform: T) -> TypedPipeline<I, T::Output>
    where
        T: Transform<Input = O> + 'static,
        T::Output: Send + Sync + 'static,
    {
        let second: Arc<dyn PipelineStep<O, T::Output>> = Arc::new(TransformStep { transform });
        TypedPipeline {
            step: Arc::new(ComposedStep {
                first: self.step,
                second,
            }),
        }
    }

    /// Apply the pipeline to a single input value.
    pub fn apply(&self, input: I) -> Result<O> {
        self.step.apply(input)
    }

    /// Apply the pipeline to a batch of inputs, returning all results.
    /// Individual errors are collected; the entire batch does **not** fail if
    /// one element fails – those elements are skipped with `Err` propagated.
    pub fn apply_batch(&self, inputs: Vec<I>) -> Result<Vec<O>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors: Vec<String> = Vec::new();
        for item in inputs {
            match self.step.apply(item) {
                Ok(out) => outputs.push(out),
                Err(e) => errors.push(e.to_string()),
            }
        }
        if !errors.is_empty() {
            // Return partial results as an error summary so callers know
            // some items failed, but include how many succeeded.
            return Err(IoError::Other(format!(
                "Batch had {} error(s): {}; {} item(s) succeeded",
                errors.len(),
                errors.join("; "),
                outputs.len()
            )));
        }
        Ok(outputs)
    }

    /// Like `apply_batch` but tolerates per-element errors, collecting both
    /// successes and failures.
    pub fn apply_batch_partial(&self, inputs: Vec<I>) -> (Vec<O>, Vec<(usize, IoError)>) {
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut errors = Vec::new();
        for (idx, item) in inputs.into_iter().enumerate() {
            match self.step.apply(item) {
                Ok(out) => outputs.push(out),
                Err(e) => errors.push((idx, e)),
            }
        }
        (outputs, errors)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in transforms
// ─────────────────────────────────────────────────────────────────────────────

// ── 1. MapTransform ──────────────────────────────────────────────────────────

/// Apply a function to each element.
pub struct MapTransform<I, O, F>
where
    F: Fn(I) -> O + Send + Sync,
{
    func: F,
    label: String,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> MapTransform<I, O, F>
where
    F: Fn(I) -> O + Send + Sync,
{
    /// Create a new `MapTransform` with the given closure.
    pub fn new(func: F) -> Self {
        Self {
            func,
            label: "map".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name for diagnostics.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<I, O, F> Transform for MapTransform<I, O, F>
where
    I: Send + Sync,
    O: Send + Sync,
    F: Fn(I) -> O + Send + Sync,
{
    type Input = I;
    type Output = O;

    fn transform(&self, input: I) -> Result<O> {
        Ok((self.func)(input))
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ── 2. FilterTransform ───────────────────────────────────────────────────────

/// Keep elements that satisfy a predicate; filtered elements produce an `Err`.
///
/// When used in `apply_batch_partial`, filtered elements appear in the error
/// list (with `IoError::Other("filtered")`).
pub struct FilterTransform<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    predicate: F,
    label: String,
    _phantom: PhantomData<T>,
}

impl<T, F> FilterTransform<T, F>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    /// Create a new `FilterTransform`.
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            label: "filter".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name for diagnostics.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<T, F> Transform for FilterTransform<T, F>
where
    T: Send + Sync,
    F: Fn(&T) -> bool + Send + Sync,
{
    type Input = T;
    type Output = T;

    fn transform(&self, input: T) -> Result<T> {
        if (self.predicate)(&input) {
            Ok(input)
        } else {
            Err(IoError::Other("filtered".to_string()))
        }
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ── 3. FlatMapTransform ──────────────────────────────────────────────────────

/// One-to-many transformation: each input produces a `Vec<O>`.
pub struct FlatMapTransform<I, O, F>
where
    F: Fn(I) -> Result<Vec<O>> + Send + Sync,
{
    func: F,
    label: String,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, F> FlatMapTransform<I, O, F>
where
    F: Fn(I) -> Result<Vec<O>> + Send + Sync,
{
    /// Create a new `FlatMapTransform`.
    pub fn new(func: F) -> Self {
        Self {
            func,
            label: "flat_map".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<I, O, F> Transform for FlatMapTransform<I, O, F>
where
    I: Send + Sync,
    O: Send + Sync,
    F: Fn(I) -> Result<Vec<O>> + Send + Sync,
{
    type Input = I;
    type Output = Vec<O>;

    fn transform(&self, input: I) -> Result<Vec<O>> {
        (self.func)(input)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ── 4. WindowTransform ───────────────────────────────────────────────────────

/// Collect a sequence into overlapping or non-overlapping sliding windows.
///
/// The transform takes a `Vec<T>` and returns `Vec<Vec<T>>` (one window per
/// entry), using cloneable items.
pub struct WindowTransform<T> {
    window_size: usize,
    step: usize,
    label: String,
    _phantom: PhantomData<T>,
}

impl<T: Clone + Send + Sync + 'static> WindowTransform<T> {
    /// Create a sliding-window transform.
    ///
    /// - `window_size`: number of items per window.
    /// - `step`: how many items to advance between windows (1 = overlapping,
    ///   `window_size` = non-overlapping / tumbling).
    pub fn new(window_size: usize, step: usize) -> Self {
        Self {
            window_size,
            step: step.max(1),
            label: "window".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<T: Clone + Send + Sync + 'static> Transform for WindowTransform<T> {
    type Input = Vec<T>;
    type Output = Vec<Vec<T>>;

    fn transform(&self, input: Vec<T>) -> Result<Vec<Vec<T>>> {
        if self.window_size == 0 {
            return Err(IoError::Other(
                "WindowTransform: window_size must be > 0".to_string(),
            ));
        }
        let windows = input
            .windows(self.window_size)
            .step_by(self.step)
            .map(|w| w.to_vec())
            .collect();
        Ok(windows)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ── 5. AggregateTransform ────────────────────────────────────────────────────

/// Group a sequence of `(key, value)` pairs and aggregate per group.
///
/// The aggregator closure receives all values for a key and returns a single
/// aggregated value.
pub struct AggregateTransform<K, V, A, F>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
    A: Clone + Send + Sync,
    F: Fn(Vec<V>) -> A + Send + Sync,
{
    aggregator: F,
    label: String,
    _phantom: PhantomData<(K, V, A)>,
}

impl<K, V, A, F> AggregateTransform<K, V, A, F>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
    A: Clone + Send + Sync,
    F: Fn(Vec<V>) -> A + Send + Sync,
{
    /// Create an `AggregateTransform`.
    ///
    /// The `aggregator` closure receives all values for a key and should
    /// return the aggregated result.
    pub fn new(aggregator: F) -> Self {
        Self {
            aggregator,
            label: "aggregate".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<K, V, A, F> Transform for AggregateTransform<K, V, A, F>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    A: Clone + Send + Sync + 'static,
    F: Fn(Vec<V>) -> A + Send + Sync + 'static,
{
    type Input = Vec<(K, V)>;
    type Output = HashMap<K, A>;

    fn transform(&self, input: Vec<(K, V)>) -> Result<HashMap<K, A>> {
        let mut groups: HashMap<K, Vec<V>> = HashMap::new();
        for (k, v) in input {
            groups.entry(k).or_default().push(v);
        }
        let result = groups
            .into_iter()
            .map(|(k, vs)| {
                let agg = (self.aggregator)(vs);
                (k, agg)
            })
            .collect();
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ── 6. JoinTransform ─────────────────────────────────────────────────────────

/// Join two sequences on a shared key.
///
/// Both sequences must be provided together as `(Vec<(K, L)>, Vec<(K, R)>)`.
/// Performs an inner join; keys present only in one sequence are dropped.
pub struct JoinTransform<K, L, R> {
    label: String,
    _phantom: PhantomData<(K, L, R)>,
}

impl<K, L, R> JoinTransform<K, L, R>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
    R: Clone + Send + Sync + 'static,
{
    /// Create a new `JoinTransform`.
    pub fn new() -> Self {
        Self {
            label: "join".to_string(),
            _phantom: PhantomData,
        }
    }

    /// Attach a descriptive name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }
}

impl<K, L, R> Default for JoinTransform<K, L, R>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
    R: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, L, R> Transform for JoinTransform<K, L, R>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
    R: Clone + Send + Sync + 'static,
{
    type Input = (Vec<(K, L)>, Vec<(K, R)>);
    type Output = Vec<(K, L, R)>;

    fn transform(&self, input: (Vec<(K, L)>, Vec<(K, R)>)) -> Result<Vec<(K, L, R)>> {
        let (left, right) = input;
        // Build a lookup from the right side (keeps last value per key for
        // duplicates – simplest semantics; full multi-join can be layered on).
        let right_map: HashMap<K, R> = right.into_iter().collect();
        let mut result = Vec::new();
        for (k, l) in left {
            if let Some(r) = right_map.get(&k) {
                result.push((k, l, r.clone()));
            }
        }
        Ok(result)
    }

    fn name(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timing / instrumentation wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps any [`Transform`] and records the elapsed time for each call.
pub struct TimedTransform<T: Transform> {
    inner: T,
    /// Accumulated wall-clock time across all `transform` calls.
    total_elapsed: std::sync::Mutex<Duration>,
    /// Number of calls so far.
    call_count: std::sync::atomic::AtomicU64,
}

impl<T: Transform> TimedTransform<T> {
    /// Create a new `TimedTransform` wrapping `inner`.
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            total_elapsed: std::sync::Mutex::new(Duration::ZERO),
            call_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Return accumulated elapsed time.
    pub fn total_elapsed(&self) -> Duration {
        self.total_elapsed
            .lock()
            .map(|g| *g)
            .unwrap_or(Duration::ZERO)
    }

    /// Return number of completed calls.
    pub fn call_count(&self) -> u64 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl<T: Transform> Transform for TimedTransform<T>
where
    T::Input: Send + Sync,
    T::Output: Send + Sync,
{
    type Input = T::Input;
    type Output = T::Output;

    fn transform(&self, input: T::Input) -> Result<T::Output> {
        let start = Instant::now();
        let result = self.inner.transform(input);
        let elapsed = start.elapsed();
        if let Ok(mut guard) = self.total_elapsed.lock() {
            *guard += elapsed;
        }
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_transform() {
        let t = MapTransform::new(|x: i32| x * 3);
        assert_eq!(t.transform(5).unwrap(), 15);
    }

    #[test]
    fn test_filter_pass() {
        let t = FilterTransform::new(|x: &i32| *x > 0);
        assert_eq!(t.transform(5).unwrap(), 5);
    }

    #[test]
    fn test_filter_reject() {
        let t = FilterTransform::new(|x: &i32| *x > 0);
        assert!(t.transform(-1).is_err());
    }

    #[test]
    fn test_flat_map_transform() {
        let t = FlatMapTransform::new(|x: i32| Ok(vec![x, x + 1, x + 2]));
        let result = t.transform(10).unwrap();
        assert_eq!(result, vec![10, 11, 12]);
    }

    #[test]
    fn test_window_transform() {
        let t = WindowTransform::<i32>::new(3, 1);
        let input = vec![1, 2, 3, 4, 5];
        let result = t.transform(input).unwrap();
        assert_eq!(result.len(), 3); // [1,2,3], [2,3,4], [3,4,5]
        assert_eq!(result[0], vec![1, 2, 3]);
        assert_eq!(result[2], vec![3, 4, 5]);
    }

    #[test]
    fn test_window_tumbling() {
        let t = WindowTransform::<i32>::new(2, 2);
        let input = vec![1, 2, 3, 4];
        let result = t.transform(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![1, 2]);
        assert_eq!(result[1], vec![3, 4]);
    }

    #[test]
    fn test_aggregate_transform() {
        let t = AggregateTransform::<&str, i32, i32, _>::new(|vs| vs.iter().sum());
        let input = vec![("a", 1), ("b", 2), ("a", 3)];
        let result = t.transform(input).unwrap();
        assert_eq!(*result.get("a").unwrap(), 4);
        assert_eq!(*result.get("b").unwrap(), 2);
    }

    #[test]
    fn test_join_transform() {
        let t = JoinTransform::<i32, &str, f64>::new();
        let left = vec![(1, "alice"), (2, "bob"), (3, "carol")];
        let right = vec![(1, 1.5), (3, 3.5)];
        let mut result = t.transform((left, right)).unwrap();
        result.sort_by_key(|(k, _, _)| *k);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, "alice", 1.5));
        assert_eq!(result[1], (3, "carol", 3.5));
    }

    #[test]
    fn test_pipeline_composition() {
        let pipeline = TypedPipeline::new()
            .then(MapTransform::new(|x: i32| x + 10))
            .then(MapTransform::new(|x: i32| x * 2));

        assert_eq!(pipeline.apply(5).unwrap(), 30); // (5+10)*2
    }

    #[test]
    fn test_pipeline_apply_batch() {
        let pipeline = TypedPipeline::new().then(MapTransform::new(|x: i32| x * x));
        let result = pipeline.apply_batch(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(result, vec![1, 4, 9, 16]);
    }

    #[test]
    fn test_pipeline_apply_batch_partial() {
        // Even numbers pass, odd numbers are filtered out
        let pipeline = TypedPipeline::new().then(FilterTransform::new(|x: &i32| x % 2 == 0));
        let (successes, failures) = pipeline.apply_batch_partial(vec![1, 2, 3, 4, 5]);
        assert_eq!(successes, vec![2, 4]);
        assert_eq!(failures.len(), 3); // indices 0,2,4
    }

    #[test]
    fn test_timed_transform() {
        let t = TimedTransform::new(MapTransform::new(|x: u64| x + 1));
        t.transform(10).unwrap();
        t.transform(20).unwrap();
        assert_eq!(t.call_count(), 2);
    }
}
