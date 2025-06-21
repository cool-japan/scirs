//! # Circuit Breaker and Error Recovery for Production Systems
//!
//! This module provides comprehensive error recovery mechanisms including circuit breakers,
//! retry logic, fallback strategies, and adaptive error handling for production environments.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed - operations are allowed
    Closed,
    /// Circuit is open - operations are rejected
    Open,
    /// Circuit is half-open - testing if service has recovered
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "closed"),
            CircuitState::Open => write!(f, "open"),
            CircuitState::HalfOpen => write!(f, "half-open"),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open the circuit
    pub failure_threshold: usize,
    /// Time window for failure counting
    pub failure_window: Duration,
    /// Recovery timeout before moving to half-open
    pub recovery_timeout: Duration,
    /// Success threshold to close the circuit from half-open
    pub success_threshold: usize,
    /// Maximum concurrent requests in half-open state
    pub max_half_open_requests: usize,
    /// Minimum number of requests before considering failure rate
    pub minimum_request_threshold: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 3,
            max_half_open_requests: 2,
            minimum_request_threshold: 10,
        }
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    /// Circuit breaker name
    name: String,
    /// Current state
    state: RwLock<CircuitState>,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Failure count
    failure_count: AtomicUsize,
    /// Success count (for half-open state)
    success_count: AtomicUsize,
    /// Total request count
    request_count: AtomicUsize,
    /// Concurrent half-open requests
    half_open_requests: AtomicUsize,
    /// Last failure time
    last_failure_time: Mutex<Option<Instant>>,
    /// Last state change time
    last_state_change: Mutex<Instant>,
    /// Failure history for time window
    failure_history: Mutex<Vec<Instant>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(name: String) -> Self {
        Self::with_config(name, CircuitBreakerConfig::default())
    }

    /// Create a circuit breaker with custom configuration
    pub fn with_config(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            name,
            state: RwLock::new(CircuitState::Closed),
            config,
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            request_count: AtomicUsize::new(0),
            half_open_requests: AtomicUsize::new(0),
            last_failure_time: Mutex::new(None),
            last_state_change: Mutex::new(Instant::now()),
            failure_history: Mutex::new(Vec::new()),
        }
    }

    /// Execute an operation with circuit breaker protection
    pub fn execute<F, T>(&self, operation: F) -> CoreResult<T>
    where
        F: FnOnce() -> CoreResult<T>,
    {
        // Check if we should allow the request
        if !self.should_allow_request()? {
            return Err(CoreError::ComputationError(ErrorContext::new(format!(
                "Circuit breaker '{}' is open - rejecting request",
                self.name
            ))));
        }

        self.request_count.fetch_add(1, Ordering::Relaxed);

        // If in half-open state, track concurrent requests
        let is_half_open = {
            let state = self.state.read().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new(
                    "Failed to read circuit breaker state",
                ))
            })?;
            *state == CircuitState::HalfOpen
        };

        if is_half_open {
            self.half_open_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Execute the operation
        let result = operation();

        // Update circuit breaker state based on result
        match &result {
            Ok(_) => self.record_success()?,
            Err(_) => self.record_failure()?,
        }

        if is_half_open {
            self.half_open_requests.fetch_sub(1, Ordering::Relaxed);
        }

        result
    }

    /// Check if we should allow the request
    fn should_allow_request(&self) -> CoreResult<bool> {
        let state = self.state.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker state"))
        })?;

        match *state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Ok(last_change) = self.last_state_change.lock() {
                    if last_change.elapsed() >= self.config.recovery_timeout {
                        drop(state); // Release read lock
                        self.transition_to_half_open()?;
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                let current_half_open = self.half_open_requests.load(Ordering::Relaxed);
                Ok(current_half_open < self.config.max_half_open_requests)
            }
        }
    }

    /// Record a successful operation
    fn record_success(&self) -> CoreResult<()> {
        let state = self.state.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker state"))
        })?;

        match *state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.config.success_threshold {
                    drop(state); // Release read lock
                    self.transition_to_closed()?;
                }
            }
            CircuitState::Open => {
                // Should not happen if should_allow_request works correctly
            }
        }

        Ok(())
    }

    /// Record a failed operation
    fn record_failure(&self) -> CoreResult<()> {
        // Update failure history
        {
            let mut history = self.failure_history.lock().map_err(|_| {
                CoreError::ComputationError(ErrorContext::new(
                    "Failed to acquire failure history lock",
                ))
            })?;

            let now = Instant::now();
            history.push(now);

            // Remove old failures outside the window
            let cutoff = now - self.config.failure_window;
            history.retain(|&failure_time| failure_time > cutoff);
        }

        self.failure_count.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut last_failure) = self.last_failure_time.lock() {
            *last_failure = Some(Instant::now());
        }

        let state = self.state.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker state"))
        })?;

        match *state {
            CircuitState::Closed => {
                if self.should_open_circuit()? {
                    drop(state); // Release read lock
                    self.transition_to_open()?;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open state triggers immediate opening
                drop(state); // Release read lock
                self.transition_to_open()?;
            }
            CircuitState::Open => {
                // Already open, nothing to do
            }
        }

        Ok(())
    }

    /// Check if circuit should open based on failure threshold
    fn should_open_circuit(&self) -> CoreResult<bool> {
        let request_count = self.request_count.load(Ordering::Relaxed);

        // Don't open if we haven't seen enough requests
        if request_count < self.config.minimum_request_threshold {
            return Ok(false);
        }

        let history = self.failure_history.lock().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire failure history lock"))
        })?;

        let recent_failures = history.len();
        Ok(recent_failures >= self.config.failure_threshold)
    }

    /// Transition to open state
    fn transition_to_open(&self) -> CoreResult<()> {
        let mut state = self.state.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to write circuit breaker state"))
        })?;

        *state = CircuitState::Open;

        if let Ok(mut last_change) = self.last_state_change.lock() {
            *last_change = Instant::now();
        }

        eprintln!("Circuit breaker '{}' opened due to failures", self.name);
        Ok(())
    }

    /// Transition to half-open state
    fn transition_to_half_open(&self) -> CoreResult<()> {
        let mut state = self.state.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to write circuit breaker state"))
        })?;

        *state = CircuitState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);

        if let Ok(mut last_change) = self.last_state_change.lock() {
            *last_change = Instant::now();
        }

        println!("Circuit breaker '{}' moved to half-open state", self.name);
        Ok(())
    }

    /// Transition to closed state
    fn transition_to_closed(&self) -> CoreResult<()> {
        let mut state = self.state.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to write circuit breaker state"))
        })?;

        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);

        if let Ok(mut last_change) = self.last_state_change.lock() {
            *last_change = Instant::now();
        }

        // Clear failure history
        if let Ok(mut history) = self.failure_history.lock() {
            history.clear();
        }

        println!("Circuit breaker '{}' closed - service recovered", self.name);
        Ok(())
    }

    /// Get current circuit breaker status
    pub fn status(&self) -> CoreResult<CircuitBreakerStatus> {
        let state = self.state.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker state"))
        })?;

        let failure_count = self.failure_count.load(Ordering::Relaxed);
        let success_count = self.success_count.load(Ordering::Relaxed);
        let request_count = self.request_count.load(Ordering::Relaxed);
        let half_open_requests = self.half_open_requests.load(Ordering::Relaxed);

        let last_failure_time = self
            .last_failure_time
            .lock()
            .map_err(|_| {
                CoreError::ComputationError(ErrorContext::new("Failed to read last failure time"))
            })?
            .map(|instant| SystemTime::now() - instant.elapsed());

        let last_state_change = self
            .last_state_change
            .lock()
            .map_err(|_| {
                CoreError::ComputationError(ErrorContext::new("Failed to read last state change"))
            })?
            .elapsed();

        Ok(CircuitBreakerStatus {
            name: self.name.clone(),
            state: state.clone(),
            failure_count,
            success_count,
            request_count,
            half_open_requests,
            last_failure_time,
            last_state_change,
        })
    }

    /// Reset the circuit breaker to closed state
    pub fn reset(&self) -> CoreResult<()> {
        let mut state = self.state.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to write circuit breaker state"))
        })?;

        *state = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.request_count.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);

        if let Ok(mut last_failure) = self.last_failure_time.lock() {
            *last_failure = None;
        }

        if let Ok(mut last_change) = self.last_state_change.lock() {
            *last_change = Instant::now();
        }

        if let Ok(mut history) = self.failure_history.lock() {
            history.clear();
        }

        println!("Circuit breaker '{}' manually reset", self.name);
        Ok(())
    }
}

/// Circuit breaker status information
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub request_count: usize,
    pub half_open_requests: usize,
    pub last_failure_time: Option<SystemTime>,
    pub last_state_change: Duration,
}

impl fmt::Display for CircuitBreakerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Circuit Breaker: {}", self.name)?;
        writeln!(f, "  State: {}", self.state)?;
        writeln!(f, "  Failures: {}", self.failure_count)?;
        writeln!(f, "  Successes: {}", self.success_count)?;
        writeln!(f, "  Total Requests: {}", self.request_count)?;
        writeln!(f, "  Half-open Requests: {}", self.half_open_requests)?;
        if let Some(last_failure) = self.last_failure_time {
            writeln!(f, "  Last Failure: {:?}", last_failure)?;
        }
        writeln!(f, "  Last State Change: {:?} ago", self.last_state_change)?;
        Ok(())
    }
}

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter to add to delays (0.0 to 1.0)
    pub jitter: f64,
    /// Whether to retry on specific error types
    pub retry_on: Vec<String>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: 0.1,
            retry_on: vec![
                "ComputationError".to_string(),
                "TimeoutError".to_string(),
                "IoError".to_string(),
            ],
        }
    }
}

/// Retry executor with exponential backoff and jitter
pub struct RetryExecutor {
    policy: RetryPolicy,
}

impl RetryExecutor {
    /// Create a new retry executor
    pub fn new(policy: RetryPolicy) -> Self {
        Self { policy }
    }

    /// Execute an operation with retry logic
    pub fn execute<F, T>(&self, operation: F) -> CoreResult<T>
    where
        F: Fn() -> CoreResult<T>,
    {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // Check if we should retry this error type
                    if !self.should_retry(&error) {
                        return Err(error);
                    }

                    last_error = Some(error);

                    // Don't sleep after the last attempt
                    if attempt < self.policy.max_attempts - 1 {
                        let delay = self.calculate_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        // All attempts failed
        Err(last_error.unwrap_or_else(|| {
            CoreError::ComputationError(ErrorContext::new("All retry attempts failed"))
        }))
    }

    /// Execute an operation with async retry logic
    pub async fn execute_async<F, Fut, T>(&self, operation: F) -> CoreResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = CoreResult<T>>,
    {
        let mut last_error = None;

        for attempt in 0..self.policy.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // Check if we should retry this error type
                    if !self.should_retry(&error) {
                        return Err(error);
                    }

                    last_error = Some(error);

                    // Don't sleep after the last attempt
                    if attempt < self.policy.max_attempts - 1 {
                        let delay = self.calculate_delay(attempt);
                        #[cfg(feature = "async")]
                        tokio::time::sleep(delay).await;
                        #[cfg(not(feature = "async"))]
                        let _delay = delay; // Avoid unused variable warning
                    }
                }
            }
        }

        // All attempts failed
        Err(last_error.unwrap_or_else(|| {
            CoreError::ComputationError(ErrorContext::new("All retry attempts failed"))
        }))
    }

    /// Check if we should retry for this error type
    fn should_retry(&self, error: &CoreError) -> bool {
        let error_type = match error {
            CoreError::ComputationError(_) => "ComputationError",
            CoreError::TimeoutError(_) => "TimeoutError",
            CoreError::IoError(_) => "IoError",
            CoreError::MemoryError(_) => "MemoryError",
            _ => return false, // Don't retry other error types by default
        };

        self.policy.retry_on.contains(&error_type.to_string())
    }

    /// Calculate delay for the given attempt
    fn calculate_delay(&self, attempt: usize) -> Duration {
        let base_delay_ms = self.policy.base_delay.as_millis() as f64;
        let exponential_delay = base_delay_ms * self.policy.backoff_multiplier.powi(attempt as i32);

        // Add jitter
        let jitter_range = exponential_delay * self.policy.jitter;
        let jitter = (rand::random::<f64>() - 0.5) * 2.0 * jitter_range;
        let delay_with_jitter = exponential_delay + jitter;

        // Cap at max delay
        let final_delay = delay_with_jitter.min(self.policy.max_delay.as_millis() as f64);

        Duration::from_millis(final_delay.max(0.0) as u64)
    }
}

/// Fallback strategy for when primary operations fail
pub trait FallbackStrategy<T>: Send + Sync {
    /// Execute the fallback strategy
    fn execute(&self, original_error: &CoreError) -> CoreResult<T>;

    /// Get the name of this fallback strategy
    fn name(&self) -> &str;
}

/// Simple fallback that returns a default value
pub struct DefaultValueFallback<T> {
    default_value: T,
    name: String,
}

impl<T: Clone> DefaultValueFallback<T> {
    /// Create a new default value fallback
    pub fn new(default_value: T, name: String) -> Self {
        Self {
            default_value,
            name,
        }
    }
}

impl<T: Clone + Send + Sync> FallbackStrategy<T> for DefaultValueFallback<T> {
    fn execute(&self, _original_error: &CoreError) -> CoreResult<T> {
        Ok(self.default_value.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Resilient executor that combines circuit breaker, retry, and fallback
pub struct ResilientExecutor<T> {
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    retry_executor: Option<RetryExecutor>,
    fallback_strategy: Option<Box<dyn FallbackStrategy<T>>>,
}

impl<T> ResilientExecutor<T> {
    /// Create a new resilient executor
    pub fn new() -> Self {
        Self {
            circuit_breaker: None,
            retry_executor: None,
            fallback_strategy: None,
        }
    }

    /// Add circuit breaker protection
    pub fn with_circuit_breaker(mut self, circuit_breaker: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(circuit_breaker);
        self
    }

    /// Add retry logic
    pub fn with_retry(mut self, retry_policy: RetryPolicy) -> Self {
        self.retry_executor = Some(RetryExecutor::new(retry_policy));
        self
    }

    /// Add fallback strategy
    pub fn with_fallback(mut self, fallback: Box<dyn FallbackStrategy<T>>) -> Self {
        self.fallback_strategy = Some(fallback);
        self
    }

    /// Execute an operation with all configured resilience patterns
    pub fn execute<F>(&self, operation: F) -> CoreResult<T>
    where
        F: Fn() -> CoreResult<T> + Clone,
    {
        let final_operation = || {
            if let Some(cb) = &self.circuit_breaker {
                cb.execute(operation.clone())
            } else {
                operation()
            }
        };

        let result = if let Some(retry) = &self.retry_executor {
            retry.execute(final_operation)
        } else {
            final_operation()
        };

        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                if let Some(fallback) = &self.fallback_strategy {
                    fallback.execute(&error)
                } else {
                    Err(error)
                }
            }
        }
    }
}

impl<T> Default for ResilientExecutor<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Global circuit breaker registry
static CIRCUIT_BREAKER_REGISTRY: std::sync::LazyLock<RwLock<HashMap<String, Arc<CircuitBreaker>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Get or create a circuit breaker
pub fn get_circuit_breaker(name: &str) -> CoreResult<Arc<CircuitBreaker>> {
    let registry = CIRCUIT_BREAKER_REGISTRY.read().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker registry"))
    })?;

    if let Some(cb) = registry.get(name) {
        return Ok(cb.clone());
    }

    drop(registry); // Release read lock

    let mut registry = CIRCUIT_BREAKER_REGISTRY.write().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new(
            "Failed to write circuit breaker registry",
        ))
    })?;

    // Double-check in case another thread created it
    if let Some(cb) = registry.get(name) {
        return Ok(cb.clone());
    }

    let circuit_breaker = Arc::new(CircuitBreaker::new(name.to_string()));
    registry.insert(name.to_string(), circuit_breaker.clone());

    Ok(circuit_breaker)
}

/// List all registered circuit breakers
pub fn list_circuit_breakers() -> CoreResult<Vec<CircuitBreakerStatus>> {
    let registry = CIRCUIT_BREAKER_REGISTRY.read().map_err(|_| {
        CoreError::ComputationError(ErrorContext::new("Failed to read circuit breaker registry"))
    })?;

    let mut statuses = Vec::new();
    for cb in registry.values() {
        statuses.push(cb.status()?);
    }

    Ok(statuses)
}

/// Convenience macros
/// Execute with circuit breaker protection
#[macro_export]
macro_rules! with_circuit_breaker {
    ($name:expr, $operation:expr) => {{
        let cb = $crate::error::circuit_breaker::get_circuit_breaker($name)?;
        cb.execute(|| $operation)
    }};
}

/// Execute with retry logic
#[macro_export]
macro_rules! with_retry {
    ($operation:expr) => {{
        let retry_executor = $crate::error::circuit_breaker::RetryExecutor::new(
            $crate::error::circuit_breaker::RetryPolicy::default(),
        );
        retry_executor.execute(|| $operation)
    }};
    ($policy:expr, $operation:expr) => {{
        let retry_executor = $crate::error::circuit_breaker::RetryExecutor::new($policy);
        retry_executor.execute(|| $operation)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_states() {
        let cb = CircuitBreaker::new("test".to_string());

        // Should start in closed state
        let status = cb.status().unwrap();
        assert_eq!(status.state, CircuitState::Closed);

        // Simulate failures to open circuit
        for _ in 0..10 {
            let _ = cb.execute(|| -> CoreResult<()> {
                Err(CoreError::ComputationError(ErrorContext::new("test error")))
            });
        }

        let status = cb.status().unwrap();
        assert_eq!(status.state, CircuitState::Open);
    }

    #[test]
    fn test_retry_executor() {
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1), // Short delay for testing
            ..Default::default()
        };

        let retry_executor = RetryExecutor::new(policy);
        let mut attempt_count = 0;

        let result = std::cell::RefCell::new(attempt_count);
        let execute_result = retry_executor.execute(|| {
            let mut count = result.borrow_mut();
            *count += 1;
            if *count < 3 {
                Err(CoreError::ComputationError(ErrorContext::new("retry test")))
            } else {
                Ok("success")
            }
        });
        attempt_count = *result.borrow();

        assert!(execute_result.is_ok());
        assert_eq!(execute_result.unwrap(), "success");
        assert_eq!(attempt_count, 3);
    }

    #[test]
    fn test_fallback_strategy() {
        let fallback = DefaultValueFallback::new(42, "test_fallback".to_string());
        let error = CoreError::ComputationError(ErrorContext::new("test error"));

        let result = fallback.execute(&error).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_resilient_executor() {
        let cb = Arc::new(CircuitBreaker::new("test".to_string()));
        let fallback = Box::new(DefaultValueFallback::new("fallback", "test".to_string()));

        let executor = ResilientExecutor::new()
            .with_circuit_breaker(cb)
            .with_fallback(fallback);

        let result = executor.execute(|| -> CoreResult<&str> {
            Err(CoreError::ComputationError(ErrorContext::new("test error")))
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "fallback");
    }

    #[test]
    fn test_circuit_breaker_registry() {
        let cb1 = get_circuit_breaker("test1").unwrap();
        let cb2 = get_circuit_breaker("test1").unwrap(); // Should get same instance

        assert!(Arc::ptr_eq(&cb1, &cb2));

        let cb3 = get_circuit_breaker("test2").unwrap(); // Different instance
        assert!(!Arc::ptr_eq(&cb1, &cb3));
    }
}
