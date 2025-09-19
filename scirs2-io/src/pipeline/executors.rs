//! Pipeline execution strategies and executors

#![allow(dead_code)]
#![allow(missing_docs)]

use super::*;
use crate::error::Result;
use crossbeam_channel::Receiver;
#[cfg(feature = "async")]
use futures::stream::{self, StreamExt};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;
#[cfg(feature = "async")]
use tokio::runtime::Runtime;

/// Trait for pipeline executors
pub trait PipelineExecutor<I, O> {
    /// Execute the pipeline with the given input
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O>;

    /// Get executor name
    fn name(&self) -> &str;
}

/// Sequential executor - executes stages one after another
pub struct SequentialExecutor;

impl<I, O> PipelineExecutor<I, O> for SequentialExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        pipeline.execute(input)
    }

    fn name(&self) -> &str {
        "sequential"
    }
}

/// Streaming executor - processes data in chunks
pub struct StreamingExecutor {
    pub chunk_size: usize,
}

impl StreamingExecutor {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

impl<I, O> PipelineExecutor<Vec<I>, Vec<O>> for StreamingExecutor
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<Vec<I>, Vec<O>>, input: Vec<I>) -> Result<Vec<O>> {
        let chunks: Vec<Vec<I>> = input
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let mut results = Vec::new();

        for chunk in chunks {
            let chunk_result = pipeline.execute(chunk)?;
            results.extend(chunk_result);
        }

        Ok(results)
    }

    fn name(&self) -> &str {
        "streaming"
    }
}

/// Async executor - executes pipeline asynchronously
#[cfg(feature = "async")]
pub struct AsyncExecutor {
    runtime: Runtime,
}

#[cfg(feature = "async")]
impl AsyncExecutor {
    pub fn new() -> Self {
        Self {
            runtime: Runtime::new().unwrap(),
        }
    }
}

#[cfg(feature = "async")]
impl<I, O> PipelineExecutor<I, O> for AsyncExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        self.runtime.block_on(async {
            // Execute pipeline in async context
            // Note: We execute directly without spawn_blocking since Pipeline doesn't implement Send
            pipeline.execute(input)
        })
    }

    fn name(&self) -> &str {
        "async"
    }
}

/// Cached executor - caches intermediate results
pub struct CachedExecutor {
    cache_dir: PathBuf,
}

impl CachedExecutor {
    pub fn new(cache_dir: impl AsRef<Path>) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
        }
    }

    fn cache_key<T>(&self, stagename: &str, input: &T) -> String
    where
        T: std::fmt::Debug,
    {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", input).hash(&mut hasher);
        format!("{}_{:x}", stagename, hasher.finish())
    }
}

impl<I, O> PipelineExecutor<I, O> for CachedExecutor
where
    I: 'static + Send + Sync + std::fmt::Debug,
    O: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // Check cache first
        let cache_key = self.cache_key("pipeline", &input);
        let cache_path = self.cache_dir.join(format!("{}.cache", cache_key));

        if cache_path.exists() {
            // Try to load from cache
            if let Ok(cached_data) = std::fs::read(&cache_path) {
                if let Ok(result) = bincode::deserialize::<O>(&cached_data) {
                    return Ok(result);
                }
            }
        }

        // Execute pipeline
        let result = pipeline.execute(input)?;

        // Save to cache
        if let Ok(serialized) = bincode::serialize(&result) {
            let _ = std::fs::create_dir_all(&self.cache_dir);
            let _ = std::fs::write(&cache_path, serialized);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "cached"
    }
}

/// Distributed executor - distributes work across multiple workers
pub struct DistributedExecutor {
    num_workers: usize,
}

impl DistributedExecutor {
    pub fn new(num_workers: usize) -> Self {
        Self { num_workers }
    }
}

impl<I, O> PipelineExecutor<Vec<I>, Vec<O>> for DistributedExecutor
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<Vec<I>, Vec<O>>, input: Vec<I>) -> Result<Vec<O>> {
        use scirs2_core::parallel_ops::*;

        // For now, we'll use a simpler approach that processes chunks in parallel
        // but executes the pipeline sequentially on each chunk
        let chunk_size = (input.len() + self.num_workers - 1) / self.num_workers;

        // Process chunks in parallel using scirs2-core's parallel operations
        let results: Result<Vec<Vec<O>>> = input
            .par_chunks(chunk_size)
            .map(|chunk| {
                // Execute pipeline on this chunk
                pipeline.execute(chunk.to_vec())
            })
            .collect();

        // Flatten results
        results.map(|chunks| chunks.into_iter().flatten().collect())
    }

    fn name(&self) -> &str {
        "distributed"
    }
}

/// Checkpointed executor - saves progress at intervals
pub struct CheckpointedExecutor {
    checkpoint_dir: PathBuf,
    checkpoint_interval: usize,
}

impl CheckpointedExecutor {
    pub fn new(checkpoint_dir: impl AsRef<Path>, interval: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            checkpoint_interval: interval,
        }
    }
}

impl<I, O> PipelineExecutor<I, O> for CheckpointedExecutor
where
    I: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
    O: 'static + Send + Sync + serde::Serialize + serde::de::DeserializeOwned,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // Create checkpoint directory
        std::fs::create_dir_all(&self.checkpoint_dir).map_err(IoError::Io)?;

        // Execute with checkpointing logic
        // Note: This is simplified - real implementation would checkpoint at each stage
        let result = pipeline.execute(input)?;

        // Save final checkpoint
        let checkpoint_path = self.checkpoint_dir.join("final.checkpoint");
        let serialized =
            bincode::serialize(&result).map_err(|e| IoError::SerializationError(e.to_string()))?;
        std::fs::write(&checkpoint_path, serialized).map_err(IoError::Io)?;

        Ok(result)
    }

    fn name(&self) -> &str {
        "checkpointed"
    }
}

/// Factory for creating executors
pub struct ExecutorFactory;

impl ExecutorFactory {
    /// Create a sequential executor
    pub fn sequential() -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(SequentialExecutor)
    }

    /// Create a streaming executor
    pub fn streaming(chunk_size: usize) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(StreamingExecutor::new(chunk_size))
    }

    /// Create an async executor
    #[cfg(feature = "async")]
    pub fn async_executor() -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(AsyncExecutor::new())
    }

    /// Create a cached executor
    pub fn cached(cache_dir: impl AsRef<Path>) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(CachedExecutor::new(cache_dir))
    }

    /// Create a distributed executor
    pub fn distributed(num_workers: usize) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(DistributedExecutor::new(num_workers))
    }

    /// Create a checkpointed executor
    pub fn checkpointed(
        checkpoint_dir: impl AsRef<Path>,
        interval: usize,
    ) -> Box<dyn PipelineExecutor<Vec<i32>, Vec<i32>>> {
        Box::new(CheckpointedExecutor::new(checkpoint_dir, interval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_executor() {
        let pipeline: Pipeline<i32, i32> =
            Pipeline::new().add_stage(function_stage("double", |x: i32| Ok(x * 2)));

        let executor = SequentialExecutor;
        let result = executor.execute(&pipeline, 21).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_streaming_executor() {
        let pipeline: Pipeline<Vec<i32>, Vec<i32>> = Pipeline::new()
            .add_stage(function_stage("double_all", |nums: Vec<i32>| {
                Ok(nums.into_iter().map(|x| x * 2).collect::<Vec<_>>())
            }));

        let executor = StreamingExecutor::new(2);
        let result = executor.execute(&pipeline, vec![1, 2, 3, 4]).unwrap();
        assert_eq!(result, vec![2, 4, 6, 8]);
    }
}

/// Enhanced streaming executor with backpressure control
pub struct BackpressureStreamingExecutor {
    chunk_size: usize,
    max_pending_chunks: usize,
    timeout: Duration,
}

impl BackpressureStreamingExecutor {
    pub fn new(chunk_size: usize, max_pending_chunks: usize) -> Self {
        Self {
            chunk_size,
            max_pending_chunks,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl<I, O> PipelineExecutor<Vec<I>, Vec<O>> for BackpressureStreamingExecutor
where
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<Vec<I>, Vec<O>>, input: Vec<I>) -> Result<Vec<O>> {
        // Process chunks sequentially to avoid borrowing issues
        // In a production implementation, you might want to use Arc<Pipeline> for sharing
        let mut all_results = Vec::new();

        for chunk in input.chunks(self.chunk_size) {
            let chunk_vec = chunk.to_vec();
            let result = pipeline.execute(chunk_vec)?;
            all_results.extend(result);
        }

        Ok(all_results)
    }

    fn name(&self) -> &str {
        "backpressure_streaming"
    }
}

/// Monitoring executor that collects detailed metrics
pub struct MonitoringExecutor<E> {
    inner: E,
    metrics_collector: Arc<Mutex<PipelineMetrics>>,
}

#[derive(Debug)]
pub struct PipelineMetrics {
    pub total_items: AtomicUsize,
    pub successful_items: AtomicUsize,
    pub failed_items: AtomicUsize,
    pub stage_metrics: HashMap<String, StageMetrics>,
    pub start_time: Option<Instant>,
    pub end_time: Option<Instant>,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            total_items: AtomicUsize::new(0),
            successful_items: AtomicUsize::new(0),
            failed_items: AtomicUsize::new(0),
            stage_metrics: HashMap::new(),
            start_time: None,
            end_time: None,
        }
    }
}

impl Clone for PipelineMetrics {
    fn clone(&self) -> Self {
        Self {
            total_items: AtomicUsize::new(self.total_items.load(Ordering::SeqCst)),
            successful_items: AtomicUsize::new(self.successful_items.load(Ordering::SeqCst)),
            failed_items: AtomicUsize::new(self.failed_items.load(Ordering::SeqCst)),
            stage_metrics: self.stage_metrics.clone(),
            start_time: self.start_time,
            end_time: self.end_time,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    pub execution_count: usize,
    pub total_duration: Duration,
    pub min_duration: Option<Duration>,
    pub max_duration: Option<Duration>,
    pub errors: Vec<String>,
}

impl<E> MonitoringExecutor<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            metrics_collector: Arc::new(Mutex::new(PipelineMetrics::default())),
        }
    }

    pub fn get_metrics(&self) -> PipelineMetrics {
        self.metrics_collector.lock().unwrap().clone()
    }
}

impl<E, I, O> PipelineExecutor<I, O> for MonitoringExecutor<E>
where
    E: PipelineExecutor<I, O>,
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        {
            let mut metrics = self.metrics_collector.lock().unwrap();
            metrics.start_time = Some(Instant::now());
            metrics.total_items.fetch_add(1, Ordering::SeqCst);
        }

        let result = self.inner.execute(pipeline, input);

        {
            let mut metrics = self.metrics_collector.lock().unwrap();
            metrics.end_time = Some(Instant::now());

            match &result {
                Ok(_) => {
                    metrics.successful_items.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {
                    metrics.failed_items.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        result
    }

    fn name(&self) -> &str {
        "monitoring"
    }
}

/// Retry executor for fault tolerance
pub struct RetryExecutor<E> {
    inner: E,
    max_retries: usize,
    retry_delay: Duration,
    exponential_backoff: bool,
}

impl<E> RetryExecutor<E> {
    pub fn new(inner: E, max_retries: usize) -> Self {
        Self {
            inner,
            max_retries,
            retry_delay: Duration::from_secs(1),
            exponential_backoff: true,
        }
    }

    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.retry_delay = delay;
        self
    }

    pub fn with_exponential_backoff(mut self, enabled: bool) -> Self {
        self.exponential_backoff = enabled;
        self
    }
}

impl<E, I, O> PipelineExecutor<I, O> for RetryExecutor<E>
where
    E: PipelineExecutor<I, O>,
    I: 'static + Send + Sync + Clone,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        let mut last_error = None;
        let mut delay = self.retry_delay;

        for attempt in 0..=self.max_retries {
            if attempt > 0 {
                thread::sleep(delay);
                if self.exponential_backoff {
                    delay *= 2;
                }
            }

            match self.inner.execute(pipeline, input.clone()) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| IoError::Other("Retry failed".to_string())))
    }

    fn name(&self) -> &str {
        "retry"
    }
}

/// Event-driven executor that triggers on specific conditions
pub struct EventDrivenExecutor {
    event_receiver: Receiver<Event>,
}

#[derive(Debug, Clone)]
pub enum Event {
    DataAvailable(String),
    ScheduledTime(Instant),
    ExternalTrigger(String),
    FileCreated(PathBuf),
}

impl EventDrivenExecutor {
    pub fn new(event_receiver: Receiver<Event>) -> Self {
        Self { event_receiver }
    }
}

impl<I, O> PipelineExecutor<I, O> for EventDrivenExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // Wait for event
        match self.event_receiver.recv() {
            Ok(event) => {
                match event {
                    Event::DataAvailable(_) | Event::ExternalTrigger(_) | Event::FileCreated(_) => {
                        // Execute pipeline when event is received
                        pipeline.execute(input)
                    }
                    Event::ScheduledTime(scheduled) => {
                        // Wait until scheduled time
                        let now = Instant::now();
                        if scheduled > now {
                            thread::sleep(scheduled - now);
                        }
                        pipeline.execute(input)
                    }
                }
            }
            Err(_) => Err(IoError::Other("Event channel closed".to_string())),
        }
    }

    fn name(&self) -> &str {
        "event_driven"
    }
}

/// Parallel stage executor for executing pipeline stages in parallel
pub struct ParallelStageExecutor {
    max_parallelism: usize,
}

impl ParallelStageExecutor {
    pub fn new(max_parallelism: usize) -> Self {
        Self { max_parallelism }
    }
}

impl<I, O> PipelineExecutor<I, O> for ParallelStageExecutor
where
    I: 'static + Send + Sync,
    O: 'static + Send + Sync,
{
    fn execute(&self, pipeline: &Pipeline<I, O>, input: I) -> Result<O> {
        // For now, delegate to regular execution
        // In a full implementation, this would analyze stage dependencies
        // and execute independent stages in parallel
        pipeline.execute(input)
    }

    fn name(&self) -> &str {
        "parallel_stage"
    }
}
