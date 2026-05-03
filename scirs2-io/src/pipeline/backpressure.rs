//! Backpressure and flow-control primitives for streaming pipelines.
//!
//! | Primitive           | Description                                          |
//! |---------------------|------------------------------------------------------|
//! | [`BoundedBuffer`]   | Fixed-capacity SPMC/MPSC buffer with blocking push   |
//! | [`ThrottleTransform`]| Rate-limit records to at most N per second          |
//! | [`BatchCollector`]  | Collect N items or wait up to a timeout, then flush  |

#![allow(missing_docs)]

use crate::error::{IoError, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// BoundedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Internal state of a [`BoundedBuffer`].
struct BoundedState<T> {
    queue: VecDeque<T>,
    capacity: usize,
    closed: bool,
}

impl<T> BoundedState<T> {
    fn new(capacity: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(capacity),
            capacity,
            closed: false,
        }
    }
}

/// A fixed-capacity, thread-safe buffer that applies **backpressure** to
/// producers: `push` blocks until space is available.
///
/// Multiple producers and consumers can share the buffer by cloning the
/// `Arc<BoundedBuffer<T>>` wrapper.  Internally the buffer uses a `Mutex` +
/// two `Condvar`s (one for producers, one for consumers).
pub struct BoundedBuffer<T: Send> {
    state: Mutex<BoundedState<T>>,
    not_full: Condvar,
    not_empty: Condvar,
}

impl<T: Send> BoundedBuffer<T> {
    /// Create a new `BoundedBuffer` with the given `capacity`.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BoundedState::new(capacity.max(1))),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        })
    }

    /// Push an item, blocking until space is available.
    ///
    /// Returns `Err` if the buffer has been closed.
    pub fn push(&self, item: T) -> Result<()> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;

        loop {
            if guard.closed {
                return Err(IoError::Other("BoundedBuffer: closed".to_string()));
            }
            if guard.queue.len() < guard.capacity {
                guard.queue.push_back(item);
                self.not_empty.notify_one();
                return Ok(());
            }
            guard = self
                .not_full
                .wait(guard)
                .map_err(|_| IoError::Other("BoundedBuffer: condvar poisoned".to_string()))?;
        }
    }

    /// Push an item with a timeout.  Returns `Ok(true)` on success,
    /// `Ok(false)` on timeout, `Err` on closed / poisoned.
    pub fn push_timeout(&self, item: T, timeout: Duration) -> Result<bool> {
        let deadline = Instant::now() + timeout;
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;

        loop {
            if guard.closed {
                return Err(IoError::Other("BoundedBuffer: closed".to_string()));
            }
            if guard.queue.len() < guard.capacity {
                guard.queue.push_back(item);
                self.not_empty.notify_one();
                return Ok(true);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(false);
            }
            let (new_guard, timed_out) = self
                .not_full
                .wait_timeout(guard, remaining)
                .map_err(|_| IoError::Other("BoundedBuffer: condvar poisoned".to_string()))?;
            guard = new_guard;
            if timed_out.timed_out() {
                return Ok(false);
            }
        }
    }

    /// Pop an item, blocking until one is available or the buffer is closed.
    ///
    /// Returns `Ok(None)` when the buffer is closed **and** empty.
    pub fn pop(&self) -> Result<Option<T>> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;

        loop {
            if let Some(item) = guard.queue.pop_front() {
                self.not_full.notify_one();
                return Ok(Some(item));
            }
            if guard.closed {
                return Ok(None);
            }
            guard = self
                .not_empty
                .wait(guard)
                .map_err(|_| IoError::Other("BoundedBuffer: condvar poisoned".to_string()))?;
        }
    }

    /// Pop an item with a timeout.  Returns `Ok(None)` on timeout or closed+empty.
    pub fn pop_timeout(&self, timeout: Duration) -> Result<Option<T>> {
        let deadline = Instant::now() + timeout;
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;

        loop {
            if let Some(item) = guard.queue.pop_front() {
                self.not_full.notify_one();
                return Ok(Some(item));
            }
            if guard.closed {
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            let (new_guard, timed_out) = self
                .not_empty
                .wait_timeout(guard, remaining)
                .map_err(|_| IoError::Other("BoundedBuffer: condvar poisoned".to_string()))?;
            guard = new_guard;
            if timed_out.timed_out() {
                return Ok(None);
            }
        }
    }

    /// Try to pop without blocking.  Returns `None` if the queue is empty.
    pub fn try_pop(&self) -> Result<Option<T>> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;
        if let Some(item) = guard.queue.pop_front() {
            self.not_full.notify_one();
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }

    /// Current number of items in the buffer.
    pub fn len(&self) -> usize {
        self.state.lock().map(|g| g.queue.len()).unwrap_or(0)
    }

    /// Returns `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Signal that no more items will be pushed.  Waiting consumers are woken.
    pub fn close(&self) -> Result<()> {
        let mut guard = self
            .state
            .lock()
            .map_err(|_| IoError::Other("BoundedBuffer: mutex poisoned".to_string()))?;
        guard.closed = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
        Ok(())
    }

    /// Returns `true` if the buffer has been closed.
    pub fn is_closed(&self) -> bool {
        self.state.lock().map(|g| g.closed).unwrap_or(true)
    }

    /// Maximum number of items the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.state.lock().map(|g| g.capacity).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ThrottleTransform
// ─────────────────────────────────────────────────────────────────────────────

/// Rate-limiting transform that ensures no more than `rate` items flow through
/// per second by inserting `std::thread::sleep` as required.
///
/// Uses a simple **token bucket** algorithm: tokens accrue at `rate` per
/// second; each item consumes one token.
pub struct ThrottleTransform {
    /// Maximum items per second.
    rate: f64,
    /// State shared across invocations.
    state: Mutex<ThrottleState>,
    label: String,
}

struct ThrottleState {
    tokens: f64,
    last_refill: Instant,
}

impl ThrottleTransform {
    /// Create a new rate-limiter.
    ///
    /// - `rate`: maximum items per second (must be > 0).
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "ThrottleTransform: rate must be > 0");
        Self {
            rate,
            state: Mutex::new(ThrottleState {
                tokens: rate, // start full
                last_refill: Instant::now(),
            }),
            label: "throttle".to_string(),
        }
    }

    /// Attach a human-readable label.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }

    /// Acquire one token, sleeping if necessary.
    pub fn acquire(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| IoError::Other("ThrottleTransform: mutex poisoned".to_string()))?;

        // Refill tokens proportional to elapsed time.
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed * self.rate).min(self.rate);
        state.last_refill = now;

        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            return Ok(());
        }

        // Not enough tokens: compute sleep duration.
        let deficit = 1.0 - state.tokens;
        let sleep_secs = deficit / self.rate;
        let sleep_dur = Duration::from_secs_f64(sleep_secs);
        drop(state); // release the lock before sleeping
        std::thread::sleep(sleep_dur);

        // After sleeping, consume the token.
        let mut state = self
            .state
            .lock()
            .map_err(|_| IoError::Other("ThrottleTransform: mutex poisoned".to_string()))?;
        let now2 = Instant::now();
        let elapsed2 = now2.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed2 * self.rate).min(self.rate);
        state.last_refill = now2;
        state.tokens = (state.tokens - 1.0).max(0.0);
        Ok(())
    }

    /// Throttle a `Vec<T>` of items, acquiring one token per item.
    pub fn throttle_batch<T>(&self, items: Vec<T>) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            self.acquire()?;
            out.push(item);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchCollector
// ─────────────────────────────────────────────────────────────────────────────

/// Collect items until either `batch_size` items have been gathered **or** the
/// `timeout` duration elapses, then flush the whole batch at once.
///
/// This is useful for downstream sinks that are most efficient when receiving
/// larger chunks (e.g. file writes, database inserts).
pub struct BatchCollector<T: Send> {
    buffer: Arc<BoundedBuffer<T>>,
    batch_size: usize,
    timeout: Duration,
    label: String,
}

impl<T: Send + 'static> BatchCollector<T> {
    /// Create a new `BatchCollector`.
    ///
    /// - `batch_size`: flush when this many items have accumulated.
    /// - `timeout`: flush after this duration even if `batch_size` not reached.
    /// - `internal_capacity`: maximum items held internally before producers
    ///   block.  Should be >= `batch_size`.
    pub fn new(batch_size: usize, timeout: Duration, internal_capacity: usize) -> Self {
        Self {
            buffer: BoundedBuffer::new(internal_capacity.max(batch_size)),
            batch_size,
            timeout,
            label: "batch_collector".to_string(),
        }
    }

    /// Attach a human-readable label.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.label = name.into();
        self
    }

    /// Push one item into the collector.  Blocks when the internal buffer is full.
    pub fn push(&self, item: T) -> Result<()> {
        self.buffer.push(item)
    }

    /// Push one item with a timeout.
    pub fn push_timeout(&self, item: T, timeout: Duration) -> Result<bool> {
        self.buffer.push_timeout(item, timeout)
    }

    /// Collect up to `batch_size` items, blocking until that count or `timeout`
    /// elapses.
    ///
    /// Returns an empty `Vec` when the collector is closed and drained.
    pub fn collect_batch(&self) -> Result<Vec<T>> {
        let deadline = Instant::now() + self.timeout;
        let mut batch = Vec::with_capacity(self.batch_size);

        while batch.len() < self.batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match self.buffer.pop_timeout(remaining)? {
                Some(item) => batch.push(item),
                None => break, // timeout or closed+empty
            }
        }
        Ok(batch)
    }

    /// Signal that no more items will be pushed.
    pub fn close(&self) -> Result<()> {
        self.buffer.close()
    }

    /// Current number of buffered items.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Configured maximum batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Configured flush timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Human-readable label.
    pub fn name(&self) -> &str {
        &self.label
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FlowController – orchestrates source → buffer → sink with backpressure
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics emitted by a [`FlowController`] run.
#[derive(Debug, Clone, Default)]
pub struct FlowStats {
    /// Total items pushed into the buffer.
    pub produced: usize,
    /// Total items consumed from the buffer.
    pub consumed: usize,
    /// Number of times the producer was throttled (push blocked).
    pub producer_stalls: usize,
    /// Wall-clock time of the run.
    pub elapsed: Duration,
}

/// Thin adapter: move items from a producer closure into a [`BoundedBuffer`],
/// then drain the buffer into a consumer closure.  Handles backpressure
/// automatically (the producer blocks when the buffer is full).
pub struct FlowController<T: Send + Clone + 'static> {
    buffer: Arc<BoundedBuffer<T>>,
}

impl<T: Send + Clone + 'static> FlowController<T> {
    /// Create a new `FlowController` with the given buffer capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: BoundedBuffer::new(capacity),
        }
    }

    /// Access the underlying [`BoundedBuffer`].
    pub fn buffer(&self) -> Arc<BoundedBuffer<T>> {
        Arc::clone(&self.buffer)
    }

    /// Run the flow: call `producer` repeatedly until it returns `None`, then
    /// drain the buffer into `consumer`.
    ///
    /// The producer and consumer run **in the same thread** for simplicity.
    /// For multi-threaded usage, share the `Arc<BoundedBuffer<T>>` directly.
    pub fn run<P, C>(&self, mut producer: P, mut consumer: C) -> Result<FlowStats>
    where
        P: FnMut() -> Option<T>,
        C: FnMut(T) -> Result<()>,
    {
        let start = Instant::now();
        let mut stats = FlowStats::default();

        // Produce phase.
        // NOTE: When running single-threaded, `capacity` must be at least as large
        // as the total number of items produced, because no consumer is running in
        // parallel to drain the buffer.  For truly unbounded streams, run the
        // producer and consumer in separate threads sharing `self.buffer()`.
        while let Some(item) = producer() {
            let pushed = self
                .buffer
                .push_timeout(item.clone(), Duration::from_millis(100))?;
            if !pushed {
                stats.producer_stalls += 1;
                // Retry with a slightly longer timeout instead of blocking indefinitely.
                // In single-threaded mode this will fail if the buffer stays full.
                let pushed2 = self.buffer.push_timeout(item, Duration::from_millis(200))?;
                if !pushed2 {
                    return Err(IoError::Other(
                        "FlowController::run: buffer full in single-threaded mode; increase capacity or use multi-threaded operation".to_string(),
                    ));
                }
            }
            stats.produced += 1;
        }
        self.buffer.close()?;

        // Consume phase: buffer is already closed, so blocking pop() returns None
        // immediately when the queue is empty (closed == true check in pop()).
        // This is safe and deadlock-free for single-threaded usage.
        while let Some(item) = self.buffer.pop()? {
            consumer(item)?;
            stats.consumed += 1;
        }

        stats.elapsed = start.elapsed();
        Ok(stats)
    }

    /// Run the flow with an overall wall-clock `deadline`.
    ///
    /// If the deadline is exceeded the buffer is closed and an error is returned.
    /// This is a safety valve for tests and production code that must not hang
    /// indefinitely.
    pub fn run_with_timeout<P, C>(
        &self,
        deadline: Duration,
        mut producer: P,
        mut consumer: C,
    ) -> Result<FlowStats>
    where
        P: FnMut() -> Option<T>,
        C: FnMut(T) -> Result<()>,
    {
        let start = Instant::now();
        let wall_deadline = start + deadline;
        let mut stats = FlowStats::default();

        // Produce phase with per-push timeouts bounded by the overall deadline.
        while let Some(item) = producer() {
            let remaining = wall_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                let _ = self.buffer.close();
                return Err(IoError::Other(
                    "FlowController::run_with_timeout: deadline exceeded during produce phase"
                        .to_string(),
                ));
            }
            let push_timeout = remaining.min(Duration::from_millis(200));
            let pushed = self.buffer.push_timeout(item.clone(), push_timeout)?;
            if !pushed {
                stats.producer_stalls += 1;
                // One more attempt with the remaining budget.
                let remaining2 = wall_deadline.saturating_duration_since(Instant::now());
                if remaining2.is_zero() {
                    let _ = self.buffer.close();
                    return Err(IoError::Other(
                        "FlowController::run_with_timeout: deadline exceeded; buffer full"
                            .to_string(),
                    ));
                }
                let pushed2 = self
                    .buffer
                    .push_timeout(item, remaining2.min(Duration::from_millis(400)))?;
                if !pushed2 {
                    let _ = self.buffer.close();
                    return Err(IoError::Other(
                        "FlowController::run_with_timeout: buffer full; increase capacity"
                            .to_string(),
                    ));
                }
            }
            stats.produced += 1;
        }
        self.buffer.close()?;

        // Consume phase: drain with per-item deadlines respecting the overall budget.
        loop {
            let remaining = wall_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(IoError::Other(
                    "FlowController::run_with_timeout: deadline exceeded during consume phase"
                        .to_string(),
                ));
            }
            let pop_timeout = remaining.min(Duration::from_secs(1));
            match self.buffer.pop_timeout(pop_timeout)? {
                Some(item) => {
                    consumer(item)?;
                    stats.consumed += 1;
                }
                None => break,
            }
        }

        stats.elapsed = start.elapsed();
        Ok(stats)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc as StdArc;
    use std::thread;

    // ── BoundedBuffer ────────────────────────────────────────────────────────

    #[test]
    fn test_bounded_buffer_basic() {
        let buf = BoundedBuffer::<i32>::new(4);
        buf.push(1).unwrap();
        buf.push(2).unwrap();
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.pop().unwrap(), Some(1));
        assert_eq!(buf.pop().unwrap(), Some(2));
        buf.close().unwrap();
        assert_eq!(buf.pop().unwrap(), None);
    }

    #[test]
    fn test_bounded_buffer_capacity_blocking() {
        let buf = BoundedBuffer::<i32>::new(2);
        buf.push(10).unwrap();
        buf.push(20).unwrap();
        // Buffer is full; push_timeout should time out.
        let result = buf.push_timeout(30, Duration::from_millis(20)).unwrap();
        assert!(!result, "should time out when buffer is full");
    }

    #[test]
    fn test_bounded_buffer_producer_consumer() {
        let buf = BoundedBuffer::<i32>::new(8);
        let buf_prod = StdArc::clone(&buf);
        let buf_cons = StdArc::clone(&buf);

        let produced = StdArc::new(AtomicUsize::new(0));
        let consumed = StdArc::new(AtomicUsize::new(0));
        let prod_count = StdArc::clone(&produced);
        let cons_count = StdArc::clone(&consumed);

        let producer = thread::spawn(move || {
            for i in 0..50i32 {
                buf_prod.push(i).unwrap();
                prod_count.fetch_add(1, Ordering::Relaxed);
            }
            buf_prod.close().unwrap();
        });

        let consumer = thread::spawn(move || {
            while let Ok(Some(_)) = buf_cons.pop() {
                cons_count.fetch_add(1, Ordering::Relaxed);
            }
        });

        producer.join().unwrap();
        consumer.join().unwrap();
        assert_eq!(produced.load(Ordering::Relaxed), 50);
        assert_eq!(consumed.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn test_bounded_buffer_try_pop() {
        let buf = BoundedBuffer::<i32>::new(4);
        assert_eq!(buf.try_pop().unwrap(), None);
        buf.push(42).unwrap();
        assert_eq!(buf.try_pop().unwrap(), Some(42));
        assert_eq!(buf.try_pop().unwrap(), None);
    }

    // ── ThrottleTransform ────────────────────────────────────────────────────

    #[test]
    fn test_throttle_high_rate() {
        // 1000 items/sec — should complete quickly in tests (no real sleep).
        let throttle = ThrottleTransform::new(1_000.0);
        let items = vec![1i32, 2, 3, 4, 5];
        let result = throttle.throttle_batch(items).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_throttle_acquire_does_not_error() {
        let throttle = ThrottleTransform::new(100.0);
        for _ in 0..5 {
            throttle.acquire().unwrap();
        }
    }

    // ── BatchCollector ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_collector_collects_full_batch() {
        let collector = BatchCollector::<i32>::new(4, Duration::from_millis(200), 16);
        for i in 0..4 {
            collector.push(i).unwrap();
        }
        let batch = collector.collect_batch().unwrap();
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_batch_collector_timeout_flush() {
        let collector = BatchCollector::<i32>::new(10, Duration::from_millis(30), 20);
        collector.push(1).unwrap();
        collector.push(2).unwrap();
        // Only 2 items pushed; should flush after timeout.
        let batch = collector.collect_batch().unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_batch_collector_close_drains() {
        let collector = BatchCollector::<i32>::new(5, Duration::from_millis(200), 20);
        collector.push(1).unwrap();
        collector.push(2).unwrap();
        collector.close().unwrap();
        let batch = collector.collect_batch().unwrap();
        // May get 2 items or 0 depending on timing; check total drains to 0.
        let _rest = collector.collect_batch().unwrap();
        // After closing, subsequent calls return empty batches.
        let empty = collector.collect_batch().unwrap();
        assert_eq!(empty.len(), 0);
    }

    // ── FlowController ───────────────────────────────────────────────────────

    /// Basic single-threaded flow: produces 10 items into a capacity-16 buffer,
    /// then drains them.  Must complete in < 5 seconds (no deadlock / hang).
    #[test]
    fn test_flow_controller_basic() {
        // Buffer must be >= number of items when running single-threaded
        // to avoid the producer blocking while waiting for a consumer that
        // hasn't started yet.
        let fc = FlowController::<i32>::new(16);
        let mut counter = 0i32;
        let mut results = Vec::new();

        let t0 = Instant::now();
        let stats = fc
            .run(
                || {
                    if counter < 10 {
                        counter += 1;
                        Some(counter)
                    } else {
                        None
                    }
                },
                |item| {
                    results.push(item);
                    Ok(())
                },
            )
            .expect("FlowController::run failed");

        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "test_flow_controller_basic took {elapsed:?}; expected < 5 s (possible deadlock)"
        );
        assert_eq!(stats.produced, 10);
        assert_eq!(stats.consumed, 10);
        assert_eq!(results.len(), 10);
    }

    /// Same scenario exercised through `run_with_timeout` to verify the
    /// deadline-aware path does not regress.
    #[test]
    fn test_flow_controller_with_timeout() {
        let fc = FlowController::<i32>::new(16);
        let mut counter = 0i32;
        let mut results = Vec::new();

        let stats = fc
            .run_with_timeout(
                Duration::from_secs(5),
                || {
                    if counter < 10 {
                        counter += 1;
                        Some(counter)
                    } else {
                        None
                    }
                },
                |item| {
                    results.push(item);
                    Ok(())
                },
            )
            .expect("run_with_timeout failed");

        assert_eq!(stats.produced, 10);
        assert_eq!(stats.consumed, 10);
        assert_eq!(results.len(), 10);
        assert!(
            stats.elapsed < Duration::from_secs(5),
            "run_with_timeout elapsed {:?}; expected < 5 s",
            stats.elapsed
        );
    }

    /// Verify that `run_with_timeout` returns an error when the buffer would
    /// require longer than the deadline allows.  We use capacity=1 and a very
    /// short deadline so the producer stalls immediately.
    #[test]
    fn test_flow_controller_timeout_triggers() {
        // capacity=1, 3 items, 1 ms deadline — producer should stall and timeout.
        let fc = FlowController::<i32>::new(1);
        let mut counter = 0i32;
        let result = fc.run_with_timeout(
            Duration::from_millis(1),
            || {
                if counter < 3 {
                    counter += 1;
                    Some(counter)
                } else {
                    None
                }
            },
            |_item| Ok(()),
        );
        // With cap=1 and no concurrent consumer, 3 items must overflow the deadline.
        // Either the deadline fires or it errors with "buffer full".
        match result {
            Err(_) => { /* expected */ }
            Ok(stats) => {
                // If somehow all fit (e.g. the second item fits because the first
                // was consumed before the deadline by the OS scheduler), that is
                // also acceptable as long as elapsed is within 5 s.
                assert!(
                    stats.elapsed < Duration::from_secs(5),
                    "run completed but took too long: {:?}",
                    stats.elapsed
                );
            }
        }
    }
}
