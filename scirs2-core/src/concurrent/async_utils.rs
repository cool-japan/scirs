//! Thread-based async-style concurrency utilities.
//!
//! This module provides concurrency primitives that mirror the semantics of
//! async libraries (semaphore, rate limiter, retry) but are implemented
//! entirely with OS threads and standard synchronisation primitives — no
//! async/await runtime required.
//!
//! # Primitives
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Semaphore`] | Counting semaphore — controls concurrent access to a resource pool. |
//! | [`TokenBucketRateLimiter`] | Token-bucket rate limiter; threads block until a token is available. |
//! | [`RetryPolicy`] | Configurable retry with exponential back-off and jitter. |
//! | [`FutureExecutor`] | Simple concurrent future executor using a fixed thread pool. |

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

// ── helpers ──────────────────────────────────────────────────────────────────

fn lock_err(ctx: &'static str, e: impl std::fmt::Display) -> CoreError {
    CoreError::MutexError(
        ErrorContext::new(format!("{ctx}: mutex poisoned: {e}"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

fn wait_err(ctx: &'static str, e: impl std::fmt::Display) -> CoreError {
    CoreError::MutexError(
        ErrorContext::new(format!("{ctx}: condvar wait poisoned: {e}"))
            .with_location(ErrorLocation::new(file!(), line!())),
    )
}

// ── Semaphore ─────────────────────────────────────────────────────────────────

/// A counting semaphore.
///
/// Maintains an internal counter initialised to `permits`.  Calls to
/// [`acquire`](Semaphore::acquire) block when the counter is 0.
/// Calls to [`release`](Semaphore::release) increment the counter and wake
/// one waiting thread.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::async_utils::Semaphore;
/// use std::sync::Arc;
///
/// let sem = Arc::new(Semaphore::new(2));
/// sem.acquire().expect("acquire");
/// sem.acquire().expect("acquire");
/// sem.release(); // now one thread can proceed
/// sem.release();
/// ```
pub struct Semaphore {
    inner: Mutex<usize>,
    condvar: Condvar,
}

impl Semaphore {
    /// Create a semaphore with `permits` initial permits.
    pub fn new(permits: usize) -> Self {
        Self {
            inner: Mutex::new(permits),
            condvar: Condvar::new(),
        }
    }

    /// Acquire one permit, blocking if none are available.
    pub fn acquire(&self) -> CoreResult<()> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("Semaphore::acquire", e))?;
        loop {
            if *g > 0 {
                *g -= 1;
                return Ok(());
            }
            g = self
                .condvar
                .wait(g)
                .map_err(|e| wait_err("Semaphore::acquire", e))?;
        }
    }

    /// Acquire one permit, blocking for at most `timeout`.
    ///
    /// Returns `true` if a permit was acquired, `false` on timeout.
    pub fn acquire_timeout(&self, timeout: Duration) -> CoreResult<bool> {
        let deadline = Instant::now() + timeout;
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("Semaphore::acquire_timeout", e))?;
        loop {
            if *g > 0 {
                *g -= 1;
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let (ng, _) = self
                .condvar
                .wait_timeout(g, remaining)
                .map_err(|e| wait_err("Semaphore::acquire_timeout", e))?;
            g = ng;
        }
    }

    /// Try to acquire without blocking.  Returns `true` on success.
    pub fn try_acquire(&self) -> CoreResult<bool> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("Semaphore::try_acquire", e))?;
        if *g > 0 {
            *g -= 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Release one permit.
    pub fn release(&self) {
        if let Ok(mut g) = self.inner.lock() {
            *g += 1;
            self.condvar.notify_one();
        }
    }

    /// Release `n` permits at once.
    pub fn release_n(&self, n: usize) {
        if let Ok(mut g) = self.inner.lock() {
            *g += n;
            if n == 1 {
                self.condvar.notify_one();
            } else {
                self.condvar.notify_all();
            }
        }
    }

    /// Current available permits (informational).
    pub fn available(&self) -> usize {
        self.inner.lock().map(|g| *g).unwrap_or(0)
    }
}

// ── RAII permit guard ────────────────────────────────────────────────────────

/// An RAII guard that releases a semaphore permit when dropped.
pub struct SemaphoreGuard<'a> {
    sem: &'a Semaphore,
}

impl<'a> SemaphoreGuard<'a> {
    /// Acquire a permit and return an RAII guard.
    pub fn acquire(sem: &'a Semaphore) -> CoreResult<Self> {
        sem.acquire()?;
        Ok(Self { sem })
    }
}

impl Drop for SemaphoreGuard<'_> {
    fn drop(&mut self) {
        self.sem.release();
    }
}

// ── TokenBucketRateLimiter ───────────────────────────────────────────────────

/// Token-bucket rate limiter.
///
/// Tokens are replenished at `rate` tokens per second up to a maximum of
/// `capacity` tokens.  Calling [`acquire`](TokenBucketRateLimiter::acquire)
/// blocks the thread until a token is available.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::async_utils::TokenBucketRateLimiter;
/// use std::time::Duration;
///
/// // Allow 100 tokens/sec with burst capacity of 10.
/// let rl = TokenBucketRateLimiter::new(10.0, 100.0);
/// rl.acquire().expect("acquire token");
/// ```
pub struct TokenBucketRateLimiter {
    inner: Mutex<TokenBucketState>,
    condvar: Condvar,
    /// Tokens per second.
    rate: f64,
    /// Maximum tokens.
    capacity: f64,
}

struct TokenBucketState {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucketRateLimiter {
    /// Create a rate limiter.
    ///
    /// - `capacity` — maximum burst size (tokens).
    /// - `rate` — sustained rate in tokens per second.
    pub fn new(capacity: f64, rate: f64) -> Self {
        let capacity = capacity.max(1.0);
        let rate = rate.max(f64::MIN_POSITIVE);
        Self {
            inner: Mutex::new(TokenBucketState {
                tokens: capacity,
                last_refill: Instant::now(),
            }),
            condvar: Condvar::new(),
            rate,
            capacity,
        }
    }

    /// Refill tokens based on elapsed time (called while holding the lock).
    fn refill(state: &mut TokenBucketState, rate: f64, capacity: f64) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed * rate).min(capacity);
        state.last_refill = now;
    }

    /// Acquire one token, blocking until one is available.
    pub fn acquire(&self) -> CoreResult<()> {
        self.acquire_n(1.0)
    }

    /// Acquire `n` tokens, blocking until they are all available.
    pub fn acquire_n(&self, n: f64) -> CoreResult<()> {
        let n = n.max(0.0);
        loop {
            let wait_duration = {
                let mut g = self
                    .inner
                    .lock()
                    .map_err(|e| lock_err("TokenBucketRateLimiter::acquire_n", e))?;
                Self::refill(&mut g, self.rate, self.capacity);
                if g.tokens >= n {
                    g.tokens -= n;
                    return Ok(());
                }
                // Compute how long until enough tokens are available.
                let deficit = n - g.tokens;
                Duration::from_secs_f64(deficit / self.rate)
            };
            thread::sleep(wait_duration.min(Duration::from_millis(50)));
        }
    }

    /// Try to acquire one token without blocking.
    ///
    /// Returns `true` if a token was acquired.
    pub fn try_acquire(&self) -> CoreResult<bool> {
        let mut g = self
            .inner
            .lock()
            .map_err(|e| lock_err("TokenBucketRateLimiter::try_acquire", e))?;
        Self::refill(&mut g, self.rate, self.capacity);
        if g.tokens >= 1.0 {
            g.tokens -= 1.0;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Current token count (approximate).
    pub fn available_tokens(&self) -> f64 {
        self.inner.lock().map(|g| g.tokens).unwrap_or(0.0)
    }
}

// ── RetryPolicy ───────────────────────────────────────────────────────────────

/// Back-off strategy for retries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStrategy {
    /// Constant delay between attempts.
    Constant,
    /// Delay doubles on each attempt (exponential back-off).
    Exponential,
    /// Exponential back-off plus ±25% random jitter.
    ExponentialWithJitter,
}

/// Configurable retry policy with back-off.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::async_utils::{RetryPolicy, BackoffStrategy};
/// use std::time::Duration;
///
/// let policy = RetryPolicy::new(5, Duration::from_millis(10), BackoffStrategy::Exponential);
/// let mut attempts = 0u32;
/// let result = policy.retry(|| {
///     attempts += 1;
///     if attempts < 3 { Err("not yet") } else { Ok(42u32) }
/// });
/// assert_eq!(result.expect("should succeed"), 42);
/// ```
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first).
    pub max_attempts: u32,
    /// Initial delay between attempts.
    pub initial_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Back-off strategy.
    pub strategy: BackoffStrategy,
}

impl RetryPolicy {
    /// Create a new retry policy.
    pub fn new(max_attempts: u32, initial_delay: Duration, strategy: BackoffStrategy) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            initial_delay,
            max_delay: Duration::from_secs(60),
            strategy,
        }
    }

    /// Override the maximum delay cap.
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Execute `f`, retrying on `Err` according to the policy.
    ///
    /// Returns the first `Ok` result, or the last `Err` if all attempts fail.
    pub fn retry<T, E, F>(&self, mut f: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut last_err = None;
        for attempt in 0..self.max_attempts {
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < self.max_attempts {
                        let delay = self.compute_delay(attempt);
                        thread::sleep(delay);
                    }
                }
            }
        }
        // SAFETY: last_err is set in the loop above (max_attempts >= 1).
        Err(last_err.expect("retry: loop did not execute"))
    }

    /// Execute `f` with a per-attempt timeout.
    pub fn retry_with_timeout<T, E, F>(&self, total_timeout: Duration, mut f: F) -> CoreResult<T>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Display,
    {
        let deadline = Instant::now() + total_timeout;
        let mut last_msg = String::new();
        for attempt in 0..self.max_attempts {
            if Instant::now() >= deadline {
                return Err(CoreError::TimeoutError(ErrorContext::new(format!(
                    "RetryPolicy: total timeout exceeded after {attempt} attempts. Last error: {last_msg}"
                ))));
            }
            match f() {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last_msg = e.to_string();
                    if attempt + 1 < self.max_attempts {
                        let delay = self
                            .compute_delay(attempt)
                            .min(deadline.saturating_duration_since(Instant::now()));
                        if !delay.is_zero() {
                            thread::sleep(delay);
                        }
                    }
                }
            }
        }
        Err(CoreError::ComputationError(ErrorContext::new(format!(
            "RetryPolicy: all {max} attempts failed. Last error: {last_msg}",
            max = self.max_attempts,
        ))))
    }

    fn compute_delay(&self, attempt: u32) -> Duration {
        let base = match self.strategy {
            BackoffStrategy::Constant => self.initial_delay,
            BackoffStrategy::Exponential | BackoffStrategy::ExponentialWithJitter => {
                let factor = 1u64.checked_shl(attempt.min(30)).unwrap_or(u64::MAX);
                self.initial_delay.saturating_mul(factor as u32)
            }
        };

        let delay = base.min(self.max_delay);

        if self.strategy == BackoffStrategy::ExponentialWithJitter {
            // Add ±25% jitter using a simple LCG.
            let seed = Instant::now().elapsed().subsec_nanos() as u64;
            let pseudo_rand = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let jitter_pct = (pseudo_rand % 50) as f64 / 100.0 - 0.25; // -0.25..+0.25
            let jitter_ns = (delay.as_nanos() as f64 * jitter_pct) as i64;
            let ns = delay.as_nanos() as i64 + jitter_ns;
            Duration::from_nanos(ns.max(0) as u64)
        } else {
            delay
        }
    }
}

// ── FutureExecutor ────────────────────────────────────────────────────────────

/// A simple concurrent future executor backed by a fixed thread pool.
///
/// Accepts closures that return `T`, executes them in parallel and returns
/// a handle that can block for the result.
///
/// # Example
///
/// ```rust
/// use scirs2_core::concurrent::async_utils::FutureExecutor;
///
/// let exec = FutureExecutor::new(4);
/// let h1 = exec.spawn(|| 2u64 + 2).expect("spawn");
/// let h2 = exec.spawn(|| 3u64 * 3).expect("spawn");
/// assert_eq!(h1.join().expect("h1"), 4);
/// assert_eq!(h2.join().expect("h2"), 9);
/// exec.shutdown().expect("shutdown");
/// ```
pub struct FutureExecutor {
    tx: Arc<Mutex<std::collections::VecDeque<(u64, Box<dyn FnOnce() + Send + 'static>)>>>,
    cond: Arc<Condvar>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
    next_id: AtomicU64,
}

impl FutureExecutor {
    /// Create an executor with `n_workers` threads.
    pub fn new(n_workers: usize) -> Self {
        let n = n_workers.max(1);
        let tx: Arc<Mutex<std::collections::VecDeque<(u64, Box<dyn FnOnce() + Send + 'static>)>>> =
            Arc::new(Mutex::new(std::collections::VecDeque::new()));
        let cond = Arc::new(Condvar::new());
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut handles = Vec::with_capacity(n);

        for _ in 0..n {
            let tx2 = Arc::clone(&tx);
            let cond2 = Arc::clone(&cond);
            let stop2 = Arc::clone(&stop);
            let handle = thread::Builder::new()
                .name("future-exec-worker".into())
                .spawn(move || loop {
                    let task = {
                        let mut g = tx2.lock().expect("executor queue lock");
                        loop {
                            if let Some(t) = g.pop_front() {
                                break Some(t);
                            }
                            if stop2.load(Ordering::Relaxed) {
                                break None;
                            }
                            g = cond2.wait(g).expect("executor condvar wait");
                        }
                    };
                    match task {
                        Some((_, f)) => f(),
                        None => break,
                    }
                })
                .expect("spawn executor worker");
            handles.push(handle);
        }

        Self {
            tx,
            cond,
            stop,
            handles,
            next_id: AtomicU64::new(0),
        }
    }

    /// Spawn a closure and return a [`JoinFuture`] that can block for the result.
    pub fn spawn<T, F>(&self, f: F) -> CoreResult<JoinFuture<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let result: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));
        let cond = Arc::new(Condvar::new());
        let result2 = Arc::clone(&result);
        let cond2 = Arc::clone(&cond);

        let task = Box::new(move || {
            let v = f();
            if let Ok(mut g) = result2.lock() {
                *g = Some(v);
                cond2.notify_one();
            }
        });

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut q = self
            .tx
            .lock()
            .map_err(|e| lock_err("FutureExecutor::spawn", e))?;
        q.push_back((id, task));
        self.cond.notify_one();

        Ok(JoinFuture { result, cond })
    }

    /// Shut down the executor.  Waits for all queued tasks to complete.
    pub fn shutdown(self) -> CoreResult<()> {
        {
            // Publish the stop flag under the queue lock so a worker that is between
            // its `stop` check and `cond.wait` cannot miss the wakeup (lost-wakeup fix).
            let _guard = self
                .tx
                .lock()
                .map_err(|e| lock_err("FutureExecutor::shutdown", e))?;
            self.stop.store(true, Ordering::SeqCst);
        }
        self.cond.notify_all();
        for h in self.handles {
            h.join().map_err(|_| {
                CoreError::SchedulerError(
                    ErrorContext::new("FutureExecutor: worker panicked on shutdown")
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
        }
        Ok(())
    }
}

/// A handle to a spawned future.  Call [`join`](JoinFuture::join) to block
/// until the value is ready.
pub struct JoinFuture<T> {
    result: Arc<Mutex<Option<T>>>,
    cond: Arc<Condvar>,
}

impl<T> JoinFuture<T> {
    /// Block until the value is ready.
    pub fn join(self) -> CoreResult<T> {
        let mut g = self
            .result
            .lock()
            .map_err(|e| lock_err("JoinFuture::join", e))?;
        loop {
            if g.is_some() {
                return g.take().ok_or_else(|| {
                    CoreError::MutexError(ErrorContext::new("JoinFuture: value already taken"))
                });
            }
            g = self
                .cond
                .wait(g)
                .map_err(|e| wait_err("JoinFuture::join", e))?;
        }
    }

    /// Block with a timeout.
    pub fn join_timeout(self, timeout: Duration) -> CoreResult<Option<T>> {
        let deadline = Instant::now() + timeout;
        let mut g = self
            .result
            .lock()
            .map_err(|e| lock_err("JoinFuture::join_timeout", e))?;
        loop {
            if g.is_some() {
                return Ok(g.take());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let (ng, _) = self
                .cond
                .wait_timeout(g, remaining)
                .map_err(|e| wait_err("JoinFuture::join_timeout", e))?;
            g = ng;
        }
    }

    /// Poll without blocking.  Returns `Some(T)` if ready, `None` otherwise.
    pub fn try_join(&self) -> CoreResult<Option<T>> {
        let mut g = self
            .result
            .lock()
            .map_err(|e| lock_err("JoinFuture::try_join", e))?;
        Ok(g.take())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AO};

    // ── Semaphore ──

    #[test]
    fn semaphore_basic_acquire_release() {
        let sem = Semaphore::new(3);
        sem.acquire().expect("acq 1");
        sem.acquire().expect("acq 2");
        sem.acquire().expect("acq 3");
        assert_eq!(sem.available(), 0);
        sem.release();
        assert_eq!(sem.available(), 1);
    }

    #[test]
    fn semaphore_try_acquire() {
        let sem = Semaphore::new(1);
        assert!(sem.try_acquire().expect("try 1"));
        assert!(!sem.try_acquire().expect("try 2 (empty)"));
        sem.release();
        assert!(sem.try_acquire().expect("try 3 after release"));
    }

    #[test]
    fn semaphore_concurrent_access() {
        let sem = Arc::new(Semaphore::new(2));
        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let s = Arc::clone(&sem);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                s.acquire().expect("acquire");
                c.fetch_add(1, AO::Relaxed);
                thread::sleep(Duration::from_millis(5));
                s.release();
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert_eq!(counter.load(AO::Relaxed), 8);
    }

    #[test]
    fn semaphore_guard() {
        let sem = Semaphore::new(1);
        {
            let _guard = SemaphoreGuard::acquire(&sem).expect("guard");
            assert_eq!(sem.available(), 0);
        }
        assert_eq!(sem.available(), 1);
    }

    #[test]
    fn semaphore_acquire_timeout_succeeds() {
        let sem = Semaphore::new(1);
        let ok = sem
            .acquire_timeout(Duration::from_millis(100))
            .expect("timeout acq");
        assert!(ok);
    }

    #[test]
    fn semaphore_acquire_timeout_expires() {
        let sem = Semaphore::new(0); // no permits
        let ok = sem
            .acquire_timeout(Duration::from_millis(20))
            .expect("timeout acq");
        assert!(!ok);
    }

    // ── TokenBucketRateLimiter ──

    #[test]
    fn rate_limiter_basic() {
        // High rate → tokens available immediately.
        let rl = TokenBucketRateLimiter::new(10.0, 1000.0);
        for _ in 0..5 {
            rl.acquire().expect("acquire token");
        }
    }

    #[test]
    fn rate_limiter_try_acquire() {
        let rl = TokenBucketRateLimiter::new(2.0, 100.0);
        assert!(rl.try_acquire().expect("t1"));
        assert!(rl.try_acquire().expect("t2"));
        assert!(!rl.try_acquire().expect("t3 empty"));
    }

    // ── RetryPolicy ──

    #[test]
    fn retry_succeeds_on_nth_attempt() {
        let counter = std::sync::atomic::AtomicU32::new(0);
        let policy = RetryPolicy::new(5, Duration::from_millis(1), BackoffStrategy::Constant);
        let result: Result<u32, &str> = policy.retry(|| {
            let n = counter.fetch_add(1, AO::Relaxed);
            if n < 3 {
                Err("not yet")
            } else {
                Ok(n)
            }
        });
        assert!(result.is_ok());
    }

    #[test]
    fn retry_exhausts_all_attempts() {
        let policy = RetryPolicy::new(3, Duration::from_millis(1), BackoffStrategy::Constant);
        let result: Result<u32, &str> = policy.retry(|| Err("always fail"));
        assert!(result.is_err());
    }

    #[test]
    fn retry_exponential_backoff() {
        let policy = RetryPolicy::new(4, Duration::from_millis(1), BackoffStrategy::Exponential);
        let counter = std::sync::atomic::AtomicU32::new(0);
        let _: Result<u32, &str> = policy.retry(|| {
            counter.fetch_add(1, AO::Relaxed);
            Err("fail")
        });
        assert_eq!(counter.load(AO::Relaxed), 4);
    }

    // ── FutureExecutor ──

    #[test]
    fn future_executor_basic() {
        let exec = FutureExecutor::new(4);
        let h1 = exec.spawn(|| 2u64 + 2).expect("spawn h1");
        let h2 = exec.spawn(|| 10u64 * 10).expect("spawn h2");
        assert_eq!(h1.join().expect("join h1"), 4);
        assert_eq!(h2.join().expect("join h2"), 100);
        exec.shutdown().expect("shutdown");
    }

    #[test]
    fn future_executor_many_tasks() {
        let exec = FutureExecutor::new(4);
        let handles: Vec<_> = (0u64..50)
            .map(|i| exec.spawn(move || i * i).expect("spawn"))
            .collect();
        for (i, h) in handles.into_iter().enumerate() {
            assert_eq!(h.join().expect("join"), (i as u64) * (i as u64));
        }
        exec.shutdown().expect("shutdown");
    }

    #[test]
    fn future_join_timeout_succeeds() {
        let exec = FutureExecutor::new(2);
        let h = exec.spawn(|| 42u64).expect("spawn");
        let v = h
            .join_timeout(Duration::from_secs(5))
            .expect("timeout join");
        assert_eq!(v, Some(42));
        exec.shutdown().expect("shutdown");
    }
}
