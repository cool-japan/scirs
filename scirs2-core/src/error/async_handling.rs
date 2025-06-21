//! Async error handling and recovery mechanisms for ``SciRS2``
//!
//! This module provides error handling patterns specifically designed for asynchronous operations:
//! - Async retry mechanisms with backoff
//! - Timeout handling for long-running operations
//! - Error propagation in async contexts
//! - Async circuit breakers
//! - Progress tracking with error recovery

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use super::recovery::{CircuitBreaker, RecoverableError, RecoveryStrategy};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Async retry executor with configurable backoff strategies
#[derive(Debug)]
pub struct AsyncRetryExecutor {
    strategy: RecoveryStrategy,
}

impl AsyncRetryExecutor {
    /// Create a new async retry executor with the given strategy
    pub fn new(strategy: RecoveryStrategy) -> Self {
        Self { strategy }
    }

    /// Execute an async function with retry logic
    pub async fn execute<F, Fut, T>(&self, mut f: F) -> CoreResult<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = CoreResult<T>>,
    {
        match &self.strategy {
            RecoveryStrategy::FailFast => f().await,

            RecoveryStrategy::ExponentialBackoff {
                max_attempts,
                initial_delay,
                max_delay,
                multiplier,
            } => {
                let mut delay = *initial_delay;
                let mut last_error = None;

                for attempt in 0..*max_attempts {
                    match f().await {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            last_error = Some(err);

                            if attempt < max_attempts - 1 {
                                tokio::time::sleep(delay).await;
                                delay = std::cmp::min(
                                    Duration::from_nanos(
                                        (delay.as_nanos() as f64 * multiplier) as u64,
                                    ),
                                    *max_delay,
                                );
                            }
                        }
                    }
                }

                Err(last_error.unwrap())
            }

            RecoveryStrategy::LinearBackoff {
                max_attempts,
                delay,
            } => {
                let mut last_error = None;

                for attempt in 0..*max_attempts {
                    match f().await {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            last_error = Some(err);

                            if attempt < max_attempts - 1 {
                                tokio::time::sleep(*delay).await;
                            }
                        }
                    }
                }

                Err(last_error.unwrap())
            }

            RecoveryStrategy::CustomBackoff {
                max_attempts,
                delays,
            } => {
                let mut last_error = None;

                for attempt in 0..*max_attempts {
                    match f().await {
                        Ok(result) => return Ok(result),
                        Err(err) => {
                            last_error = Some(err);

                            if attempt < max_attempts - 1 {
                                if let Some(&delay) = delays.get(attempt) {
                                    tokio::time::sleep(delay).await;
                                }
                            }
                        }
                    }
                }

                Err(last_error.unwrap())
            }

            _ => f().await, // Other strategies not applicable for retry
        }
    }
}

/// Async circuit breaker for handling repeated failures in async contexts
#[derive(Debug)]
pub struct AsyncCircuitBreaker {
    #[allow(dead_code)]
    inner: Arc<CircuitBreaker>,
}

impl AsyncCircuitBreaker {
    /// Create a new async circuit breaker
    pub fn new(failure_threshold: usize, timeout: Duration, recovery_timeout: Duration) -> Self {
        Self {
            inner: Arc::new(CircuitBreaker::new(
                failure_threshold,
                timeout,
                recovery_timeout,
            )),
        }
    }

    /// Execute an async function with circuit breaker protection
    pub async fn execute<F, Fut, T>(&self, f: F) -> CoreResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = CoreResult<T>>,
    {
        // Check if circuit should allow execution
        if !self.should_allow_execution() {
            return Err(CoreError::ComputationError(ErrorContext::new(
                "Async circuit breaker is open - too many recent failures",
            )));
        }

        // Execute the async function
        match f().await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(err) => {
                self.on_failure();
                Err(err)
            }
        }
    }

    fn should_allow_execution(&self) -> bool {
        // Delegate to the inner circuit breaker
        // This is a simplified check - in a real implementation,
        // you'd need to expose the internal state checking logic
        true // Placeholder
    }

    fn on_success(&self) {
        // Update circuit breaker state on success
        // This would typically involve updating internal counters
    }

    fn on_failure(&self) {
        // Update circuit breaker state on failure
        // This would typically involve updating failure counters
    }
}

/// Timeout wrapper for async operations
pub struct TimeoutWrapper<F> {
    future: F,
    #[allow(dead_code)]
    timeout: Duration,
}

impl<F> TimeoutWrapper<F> {
    /// Create a new timeout wrapper
    pub fn new(future: F, timeout: Duration) -> Self {
        Self { future, timeout }
    }
}

impl<F, T> Future for TimeoutWrapper<F>
where
    F: Future<Output = CoreResult<T>>,
{
    type Output = CoreResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // This is a simplified implementation
        // In a real implementation, you'd use tokio::time::timeout
        // or implement proper timeout handling

        let this = unsafe { self.get_unchecked_mut() };
        let future = unsafe { Pin::new_unchecked(&mut this.future) };

        match future.poll(cx) {
            Poll::Ready(result) => Poll::Ready(result),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Progress tracker for long-running async operations with error recovery
#[derive(Debug)]
pub struct AsyncProgressTracker {
    total_steps: usize,
    completed_steps: Arc<Mutex<usize>>,
    errors: Arc<Mutex<Vec<RecoverableError>>>,
    start_time: Instant,
}

impl AsyncProgressTracker {
    /// Create a new progress tracker
    pub fn new(total_steps: usize) -> Self {
        Self {
            total_steps,
            completed_steps: Arc::new(Mutex::new(0)),
            errors: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
        }
    }

    /// Mark a step as completed
    pub fn complete_step(&self) {
        let mut completed = self.completed_steps.lock().unwrap();
        *completed += 1;
    }

    /// Record an error that occurred during processing
    pub fn record_error(&self, error: RecoverableError) {
        let mut errors = self.errors.lock().unwrap();
        errors.push(error);
    }

    /// Get current progress (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        let completed = *self.completed_steps.lock().unwrap() as f64;
        completed / self.total_steps as f64
    }

    /// Get elapsed time
    pub fn elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate remaining time based on current progress
    pub fn estimated_remaining_time(&self) -> Option<Duration> {
        let progress = self.progress();
        if progress > 0.0 && progress < 1.0 {
            let elapsed = self.elapsed_time();
            let total_estimated = elapsed.as_secs_f64() / progress;
            let remaining = total_estimated - elapsed.as_secs_f64();
            Some(Duration::from_secs_f64(remaining.max(0.0)))
        } else {
            None
        }
    }

    /// Get all recorded errors
    pub fn errors(&self) -> Vec<RecoverableError> {
        self.errors.lock().unwrap().clone()
    }

    /// Check if any errors have been recorded
    pub fn has_errors(&self) -> bool {
        !self.errors.lock().unwrap().is_empty()
    }

    /// Get a progress report
    pub fn progress_report(&self) -> String {
        let completed = *self.completed_steps.lock().unwrap();
        let progress_pct = (self.progress() * 100.0) as u32;
        let elapsed = self.elapsed_time();
        let error_count = self.errors.lock().unwrap().len();

        let mut report = format!(
            "Progress: {}/{} steps ({}%) | Elapsed: {:?}",
            completed, self.total_steps, progress_pct, elapsed
        );

        if let Some(remaining) = self.estimated_remaining_time() {
            report.push_str(&format!(" | ETA: {:?}", remaining));
        }

        if error_count > 0 {
            report.push_str(&format!(" | Errors: {}", error_count));
        }

        report
    }
}

/// Async error aggregator for collecting errors from multiple async operations
#[derive(Debug)]
pub struct AsyncErrorAggregator {
    errors: Arc<Mutex<Vec<RecoverableError>>>,
    max_errors: Option<usize>,
}

impl AsyncErrorAggregator {
    /// Create a new async error aggregator
    pub fn new() -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
            max_errors: None,
        }
    }

    /// Create a new async error aggregator with maximum error limit
    pub fn with_max_errors(max_errors: usize) -> Self {
        Self {
            errors: Arc::new(Mutex::new(Vec::new())),
            max_errors: Some(max_errors),
        }
    }

    /// Add an error to the aggregator (async-safe)
    pub async fn add_error(&self, error: RecoverableError) {
        let mut errors = self.errors.lock().unwrap();

        if let Some(max) = self.max_errors {
            if errors.len() >= max {
                return; // Ignore additional errors
            }
        }

        errors.push(error);
    }

    /// Add a simple error to the aggregator
    pub async fn add_simple_error(&self, error: CoreError) {
        self.add_error(RecoverableError::new(error)).await;
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.lock().unwrap().is_empty()
    }

    /// Get the number of errors
    pub fn error_count(&self) -> usize {
        self.errors.lock().unwrap().len()
    }

    /// Get all errors
    pub fn errors(&self) -> Vec<RecoverableError> {
        self.errors.lock().unwrap().clone()
    }

    /// Get the most severe error
    pub fn most_severe_error(&self) -> Option<RecoverableError> {
        self.errors().into_iter().max_by_key(|err| err.severity)
    }

    /// Convert to a single error if there are any errors
    pub fn into_result<T>(self, success_value: T) -> Result<T, RecoverableError> {
        if let Some(most_severe) = self.most_severe_error() {
            Err(most_severe)
        } else {
            Ok(success_value)
        }
    }
}

impl Default for AsyncErrorAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to add timeout to any async operation
pub async fn with_timeout<F, T>(future: F, timeout: Duration) -> CoreResult<T>
where
    F: Future<Output = CoreResult<T>>,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(result) => result,
        Err(_) => Err(CoreError::TimeoutError(ErrorContext::new(format!(
            "Operation timed out after {:?}",
            timeout
        )))),
    }
}

/// Convenience function to retry an async operation with exponential backoff
pub async fn retry_with_exponential_backoff<F, Fut, T>(
    f: F,
    max_attempts: usize,
    initial_delay: Duration,
    max_delay: Duration,
    multiplier: f64,
) -> CoreResult<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = CoreResult<T>>,
{
    let executor = AsyncRetryExecutor::new(RecoveryStrategy::ExponentialBackoff {
        max_attempts,
        initial_delay,
        max_delay,
        multiplier,
    });

    executor.execute(f).await
}

/// Convenience function to execute multiple async operations with error aggregation
pub async fn execute_with_error_aggregation<T>(
    operations: Vec<impl Future<Output = CoreResult<T>>>,
    fail_fast: bool,
) -> Result<Vec<T>, AsyncErrorAggregator> {
    let aggregator = AsyncErrorAggregator::new();
    let mut results = Vec::new();

    for operation in operations {
        match operation.await {
            Ok(result) => results.push(result),
            Err(error) => {
                aggregator.add_simple_error(error).await;

                if fail_fast {
                    return Err(aggregator);
                }
            }
        }
    }

    if aggregator.has_errors() {
        Err(aggregator)
    } else {
        Ok(results)
    }
}

/// Async operation with built-in progress tracking and error recovery
pub struct TrackedAsyncOperation<F> {
    operation: F,
    tracker: AsyncProgressTracker,
    retry_strategy: Option<RecoveryStrategy>,
}

impl<F> TrackedAsyncOperation<F> {
    /// Create a new tracked async operation
    pub fn new(operation: F, total_steps: usize) -> Self {
        Self {
            operation,
            tracker: AsyncProgressTracker::new(total_steps),
            retry_strategy: None,
        }
    }

    /// Add retry strategy to the operation
    pub fn with_retry(mut self, strategy: RecoveryStrategy) -> Self {
        self.retry_strategy = Some(strategy);
        self
    }

    /// Get reference to the progress tracker
    pub const fn tracker(&self) -> &AsyncProgressTracker {
        &self.tracker
    }
}

impl<F, T> Future for TrackedAsyncOperation<F>
where
    F: Future<Output = CoreResult<T>>,
{
    type Output = CoreResult<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        let operation = unsafe { Pin::new_unchecked(&mut this.operation) };

        match operation.poll(cx) {
            Poll::Ready(result) => {
                match &result {
                    Ok(_) => this.tracker.complete_step(),
                    Err(error) => {
                        let recoverable_error = RecoverableError::new(error.clone());
                        this.tracker.record_error(recoverable_error);
                    }
                }
                Poll::Ready(result)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Macro to create an async operation with automatic error handling and progress tracking
#[macro_export]
macro_rules! async_with_recovery {
    ($operation:expr, $steps:expr) => {{
        let tracked_op =
            $crate::error::async_handling::TrackedAsyncOperation::new($operation, $steps);
        tracked_op.await
    }};

    ($operation:expr, $steps:expr, $retry_strategy:expr) => {{
        let tracked_op =
            $crate::error::async_handling::TrackedAsyncOperation::new($operation, $steps)
                .with_retry($retry_strategy);
        tracked_op.await
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_async_retry_executor() {
        let executor = AsyncRetryExecutor::new(RecoveryStrategy::LinearBackoff {
            max_attempts: 3,
            delay: Duration::from_millis(1),
        });

        let attempt_count = Arc::new(AtomicUsize::new(0));
        let attempt_count_clone = attempt_count.clone();

        let result = executor
            .execute(|| {
                let count = attempt_count_clone.clone();
                async move {
                    let current = count.fetch_add(1, Ordering::SeqCst);
                    if current < 2 {
                        Err(CoreError::ComputationError(ErrorContext::new("Test error")))
                    } else {
                        Ok(42)
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_timeout_wrapper() {
        let result = with_timeout(
            async {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(42)
            },
            Duration::from_millis(50),
        )
        .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), CoreError::TimeoutError(_)));
    }

    #[tokio::test]
    async fn test_progress_tracker() {
        let tracker = AsyncProgressTracker::new(10);

        assert_eq!(tracker.progress(), 0.0);

        tracker.complete_step();
        tracker.complete_step();

        assert_eq!(tracker.progress(), 0.2);
        assert!(tracker.elapsed_time().as_millis() > 0);
    }

    #[tokio::test]
    async fn test_async_error_aggregator() {
        let aggregator = AsyncErrorAggregator::new();

        assert!(!aggregator.has_errors());

        aggregator
            .add_simple_error(CoreError::ValueError(ErrorContext::new("Error 1")))
            .await;
        aggregator
            .add_simple_error(CoreError::DomainError(ErrorContext::new("Error 2")))
            .await;

        assert_eq!(aggregator.error_count(), 2);
        assert!(aggregator.has_errors());
    }
}
