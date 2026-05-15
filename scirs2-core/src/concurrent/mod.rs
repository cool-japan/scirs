//! Concurrent data structures for safe shared-state parallelism.
//!
//! This module provides high-performance, lock-based concurrent data structures:
//!
//! - [`ConcurrentHashMap`] — sharded-lock hash map for high-throughput concurrent access
//! - [`BoundedQueue`] — bounded MPMC (multi-producer, multi-consumer) queue
//! - [`ConcurrentAccumulator`] — lock-free parallel accumulator for reductions
//! - [`WriterPreferenceRwLock`] — read-write lock that gives priority to writers
//! - [`DoubleBuffer`] — double-buffered exchange for producer/consumer patterns
//!
//! All structures are `Send + Sync` and avoid `unwrap()` in favour of explicit error handling.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::concurrent::{ConcurrentHashMap, BoundedQueue};
//!
//! // Concurrent hash map
//! let map: ConcurrentHashMap<String, u64> = ConcurrentHashMap::new();
//! map.insert("key".into(), 42);
//! assert_eq!(map.get(&"key".to_string()), Some(42));
//!
//! // Bounded MPMC queue
//! let queue: BoundedQueue<i32> = BoundedQueue::new(16);
//! queue.push(1).expect("should succeed");
//! assert_eq!(queue.pop(), Some(1));
//! ```

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::error::{CoreError, CoreResult, ErrorContext};

// ===========================================================================
// ConcurrentHashMap (sharded lock)
// ===========================================================================

/// Default number of shards for [`ConcurrentHashMap`].
const DEFAULT_SHARD_COUNT: usize = 64;

/// A concurrent hash map using shard-level locking for high throughput.
///
/// The map is split into `N` independent shards, each protected by its own
/// mutex.  This dramatically reduces contention compared to a single global
/// lock when many threads operate on different keys simultaneously.
pub struct ConcurrentHashMap<K, V, S = std::hash::RandomState> {
    shards: Vec<Mutex<HashMap<K, V, S>>>,
    shard_count: usize,
    hash_builder: S,
    len: AtomicUsize,
}

impl<K, V> ConcurrentHashMap<K, V, std::hash::RandomState>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new concurrent hash map with the default number of shards.
    pub fn new() -> Self {
        Self::with_shard_count(DEFAULT_SHARD_COUNT)
    }

    /// Create a concurrent hash map with the specified number of shards.
    pub fn with_shard_count(n: usize) -> Self {
        let shard_count = n.max(1);
        let hash_builder = std::hash::RandomState::new();
        let shards = (0..shard_count)
            .map(|_| Mutex::new(HashMap::with_hasher(hash_builder.clone())))
            .collect();
        Self {
            shards,
            shard_count,
            hash_builder,
            len: AtomicUsize::new(0),
        }
    }
}

impl<K, V, S> ConcurrentHashMap<K, V, S>
where
    K: Eq + Hash + Clone,
    V: Clone,
    S: BuildHasher + Clone,
{
    fn shard_index(&self, key: &K) -> usize {
        (self.hash_builder.hash_one(&key) as usize) % self.shard_count
    }

    fn lock_shard(&self, idx: usize) -> CoreResult<MutexGuard<'_, HashMap<K, V, S>>> {
        self.shards[idx].lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!(
                "concurrent hash map shard {idx} mutex poisoned: {e}"
            )))
        })
    }

    /// Insert a key-value pair, returning the previous value if the key already existed.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let idx = self.shard_index(&key);
        let mut shard = match self.lock_shard(idx) {
            Ok(s) => s,
            Err(_) => return None,
        };
        let prev = shard.insert(key, value);
        if prev.is_none() {
            self.len.fetch_add(1, Ordering::Release);
        }
        prev
    }

    /// Retrieve a clone of the value associated with the key.
    pub fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let shard = match self.lock_shard(idx) {
            Ok(s) => s,
            Err(_) => return None,
        };
        shard.get(key).cloned()
    }

    /// Remove a key and return its value.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut shard = match self.lock_shard(idx) {
            Ok(s) => s,
            Err(_) => return None,
        };
        let removed = shard.remove(key);
        if removed.is_some() {
            self.len.fetch_sub(1, Ordering::Release);
        }
        removed
    }

    /// Check whether the map contains the key.
    pub fn contains_key(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        match self.lock_shard(idx) {
            Ok(shard) => shard.contains_key(key),
            Err(_) => false,
        }
    }

    /// Approximate number of entries (may be slightly stale under contention).
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Apply a function to the value associated with `key`, returning the result.
    ///
    /// If the key doesn't exist, returns `None`.
    pub fn get_and_modify<F, R>(&self, key: &K, f: F) -> Option<R>
    where
        F: FnOnce(&mut V) -> R,
    {
        let idx = self.shard_index(key);
        let mut shard = match self.lock_shard(idx) {
            Ok(s) => s,
            Err(_) => return None,
        };
        shard.get_mut(key).map(f)
    }

    /// Insert if the key does not already exist, using a closure to create the value.
    ///
    /// Returns a clone of the existing or newly-inserted value.
    pub fn get_or_insert_with<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce() -> V,
    {
        let idx = self.shard_index(&key);
        let mut shard = match self.lock_shard(idx) {
            Ok(s) => s,
            Err(_) => return f(),
        };
        if let Some(existing) = shard.get(&key) {
            return existing.clone();
        }
        let value = f();
        let cloned = value.clone();
        shard.insert(key, value);
        self.len.fetch_add(1, Ordering::Release);
        cloned
    }

    /// Collect all keys into a Vec.
    pub fn keys(&self) -> Vec<K> {
        let mut result = Vec::new();
        for shard_mutex in &self.shards {
            if let Ok(shard) = shard_mutex.lock() {
                result.extend(shard.keys().cloned());
            }
        }
        result
    }

    /// Clear all entries.
    pub fn clear(&self) {
        for shard_mutex in &self.shards {
            if let Ok(mut shard) = shard_mutex.lock() {
                shard.clear();
            }
        }
        self.len.store(0, Ordering::Release);
    }
}

// Safety: The sharded mutexes ensure thread-safe access.
unsafe impl<K: Send, V: Send, S: Send> Send for ConcurrentHashMap<K, V, S> {}
unsafe impl<K: Send + Sync, V: Send + Sync, S: Send + Sync> Sync for ConcurrentHashMap<K, V, S> {}

impl<K, V> Default for ConcurrentHashMap<K, V, std::hash::RandomState>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// BoundedQueue (MPMC)
// ===========================================================================

/// A bounded, multi-producer, multi-consumer (MPMC) queue.
///
/// Pushing to a full queue returns an error rather than blocking.
/// Popping from an empty queue returns `None`.
/// Blocking variants (`push_blocking`, `pop_blocking`) are available.
pub struct BoundedQueue<T> {
    buffer: Mutex<std::collections::VecDeque<T>>,
    capacity: usize,
    not_empty: Condvar,
    not_full: Condvar,
    len: AtomicUsize,
    closed: AtomicBool,
}

impl<T> BoundedQueue<T> {
    /// Create a new bounded queue with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            buffer: Mutex::new(std::collections::VecDeque::with_capacity(cap)),
            capacity: cap,
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            len: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
        }
    }

    /// Try to push an item, returning `Err(item)` if the queue is full or closed.
    pub fn push(&self, item: T) -> Result<(), T> {
        if self.closed.load(Ordering::Acquire) {
            return Err(item);
        }
        let mut buf = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return Err(item),
        };
        if buf.len() >= self.capacity {
            return Err(item);
        }
        buf.push_back(item);
        self.len.fetch_add(1, Ordering::Release);
        self.not_empty.notify_one();
        Ok(())
    }

    /// Push an item, blocking until space is available or the queue is closed.
    pub fn push_blocking(&self, item: T) -> CoreResult<()> {
        self.push_blocking_timeout(item, None)
    }

    /// Push an item with an optional timeout.
    pub fn push_blocking_timeout(&self, mut item: T, timeout: Option<Duration>) -> CoreResult<()> {
        let deadline = timeout.map(|d| std::time::Instant::now() + d);

        loop {
            if self.closed.load(Ordering::Acquire) {
                return Err(CoreError::ComputationError(ErrorContext::new(
                    "queue is closed".to_string(),
                )));
            }

            let mut buf = self.buffer.lock().map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!("queue mutex poisoned: {e}")))
            })?;

            if buf.len() < self.capacity {
                buf.push_back(item);
                self.len.fetch_add(1, Ordering::Release);
                self.not_empty.notify_one();
                return Ok(());
            }

            // Wait for space.
            if let Some(dl) = deadline {
                let remaining = dl.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Err(CoreError::ComputationError(ErrorContext::new(
                        "push timed out".to_string(),
                    )));
                }
                let (b, timeout_result) =
                    self.not_full.wait_timeout(buf, remaining).map_err(|e| {
                        CoreError::ComputationError(ErrorContext::new(format!(
                            "condvar wait failed: {e}"
                        )))
                    })?;
                drop(b);
                if timeout_result.timed_out() {
                    return Err(CoreError::ComputationError(ErrorContext::new(
                        "push timed out".to_string(),
                    )));
                }
            } else {
                let _b = self.not_full.wait(buf).map_err(|e| {
                    CoreError::ComputationError(ErrorContext::new(format!(
                        "condvar wait failed: {e}"
                    )))
                })?;
            }
        }
    }

    /// Try to pop an item, returning `None` if the queue is empty.
    pub fn pop(&self) -> Option<T> {
        let mut buf = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return None,
        };
        let item = buf.pop_front();
        if item.is_some() {
            self.len.fetch_sub(1, Ordering::Release);
            self.not_full.notify_one();
        }
        item
    }

    /// Pop an item, blocking until one is available or the queue is closed.
    pub fn pop_blocking(&self) -> CoreResult<Option<T>> {
        self.pop_blocking_timeout(None)
    }

    /// Pop an item with an optional timeout.
    pub fn pop_blocking_timeout(&self, timeout: Option<Duration>) -> CoreResult<Option<T>> {
        let deadline = timeout.map(|d| std::time::Instant::now() + d);

        loop {
            let mut buf = self.buffer.lock().map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!("queue mutex poisoned: {e}")))
            })?;

            if let Some(item) = buf.pop_front() {
                self.len.fetch_sub(1, Ordering::Release);
                self.not_full.notify_one();
                return Ok(Some(item));
            }

            if self.closed.load(Ordering::Acquire) {
                return Ok(None);
            }

            if let Some(dl) = deadline {
                let remaining = dl.saturating_duration_since(std::time::Instant::now());
                if remaining.is_zero() {
                    return Ok(None); // timed out
                }
                let (b, timeout_result) =
                    self.not_empty.wait_timeout(buf, remaining).map_err(|e| {
                        CoreError::ComputationError(ErrorContext::new(format!(
                            "condvar wait failed: {e}"
                        )))
                    })?;
                drop(b);
                if timeout_result.timed_out() {
                    return Ok(None);
                }
            } else {
                let _b = self.not_empty.wait(buf).map_err(|e| {
                    CoreError::ComputationError(ErrorContext::new(format!(
                        "condvar wait failed: {e}"
                    )))
                })?;
            }
        }
    }

    /// Number of items currently in the queue.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum capacity of the queue.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Close the queue; no more pushes are accepted.
    ///
    /// Blocked consumers will wake up and receive `None`.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Whether the queue is closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

unsafe impl<T: Send> Send for BoundedQueue<T> {}
unsafe impl<T: Send> Sync for BoundedQueue<T> {}

// ===========================================================================
// ConcurrentAccumulator (for parallel reductions)
// ===========================================================================

/// A concurrent accumulator for parallel reductions.
///
/// Supports atomic `f64` accumulation via sharded slots to reduce contention,
/// plus a generic `T` accumulator that uses a mutex.
pub struct ConcurrentAccumulator<T: Clone> {
    /// Sharded accumulators to reduce contention.
    shards: Vec<Mutex<T>>,
    shard_count: usize,
    /// Combining function: `combine(accumulator, new_value) -> accumulator`.
    combiner: Arc<dyn Fn(T, T) -> T + Send + Sync>,
    /// Identity / zero element for the accumulation.
    identity: T,
    /// Counter for round-robin shard selection.
    counter: AtomicUsize,
}

impl<T: Clone + Send + Sync + 'static> ConcurrentAccumulator<T> {
    /// Create a new accumulator with the given identity element and combiner function.
    ///
    /// `shard_count` controls the number of independent accumulator slots (more = less contention).
    pub fn new<F>(identity: T, combiner: F, shard_count: usize) -> Self
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        let sc = shard_count.max(1);
        let shards = (0..sc).map(|_| Mutex::new(identity.clone())).collect();
        Self {
            shards,
            shard_count: sc,
            combiner: Arc::new(combiner),
            identity,
            counter: AtomicUsize::new(0),
        }
    }

    /// Accumulate a value into one of the shards.
    pub fn accumulate(&self, value: T) {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.shard_count;
        if let Ok(mut shard) = self.shards[idx].lock() {
            let old = shard.clone();
            *shard = (self.combiner)(old, value);
        }
    }

    /// Combine all shards and return the final accumulated value.
    pub fn result(&self) -> T {
        let mut acc = self.identity.clone();
        for shard_mutex in &self.shards {
            if let Ok(shard) = shard_mutex.lock() {
                acc = (self.combiner)(acc, shard.clone());
            }
        }
        acc
    }

    /// Reset all shards to the identity element.
    pub fn reset(&self) {
        for shard_mutex in &self.shards {
            if let Ok(mut shard) = shard_mutex.lock() {
                *shard = self.identity.clone();
            }
        }
        self.counter.store(0, Ordering::Relaxed);
    }
}

unsafe impl<T: Clone + Send> Send for ConcurrentAccumulator<T> {}
unsafe impl<T: Clone + Send + Sync> Sync for ConcurrentAccumulator<T> {}

// ---------------------------------------------------------------------------
// Specialised f64 accumulator using AtomicU64 for lock-free sum
// ---------------------------------------------------------------------------

/// A lock-free accumulator specialised for `f64` summation.
///
/// Uses atomic compare-and-swap on the bits of an `f64` for true lock-free operation.
pub struct AtomicF64Accumulator {
    bits: AtomicU64,
    count: AtomicU64,
}

impl AtomicF64Accumulator {
    /// Create a new accumulator initialised to 0.0.
    pub fn new() -> Self {
        Self {
            bits: AtomicU64::new(0.0_f64.to_bits()),
            count: AtomicU64::new(0),
        }
    }

    /// Atomically add `value` to the accumulator.
    pub fn add(&self, value: f64) {
        loop {
            let current_bits = self.bits.load(Ordering::Acquire);
            let current = f64::from_bits(current_bits);
            let new = current + value;
            let new_bits = new.to_bits();
            if self
                .bits
                .compare_exchange_weak(current_bits, new_bits, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.count.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Read the current accumulated value.
    pub fn value(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Acquire))
    }

    /// Number of values accumulated so far.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Acquire)
    }

    /// Reset to 0.0.
    pub fn reset(&self) {
        self.bits.store(0.0_f64.to_bits(), Ordering::Release);
        self.count.store(0, Ordering::Release);
    }
}

impl Default for AtomicF64Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// WriterPreferenceRwLock
// ===========================================================================

/// A read-write lock that gives priority to writers.
///
/// When a writer is waiting, new readers are blocked until the writer has been serviced.
/// This prevents writer starvation under heavy read load.
pub struct WriterPreferenceRwLock<T> {
    data: std::sync::RwLock<T>,
    /// Number of writers waiting.
    writers_waiting: AtomicUsize,
    /// Gate that readers must pass through; blocked when writers are waiting.
    reader_gate: Mutex<()>,
    reader_gate_cv: Condvar,
}

impl<T> WriterPreferenceRwLock<T> {
    /// Create a new writer-preference RwLock wrapping `data`.
    pub fn new(data: T) -> Self {
        Self {
            data: std::sync::RwLock::new(data),
            writers_waiting: AtomicUsize::new(0),
            reader_gate: Mutex::new(()),
            reader_gate_cv: Condvar::new(),
        }
    }

    /// Acquire a read lock.
    ///
    /// If a writer is waiting, this will block until the writer has been serviced.
    pub fn read(&self) -> CoreResult<ReadGuard<'_, T>> {
        // Wait until no writers are waiting.
        {
            let mut gate = self.reader_gate.lock().map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!("gate mutex poisoned: {e}")))
            })?;
            while self.writers_waiting.load(Ordering::Acquire) > 0 {
                gate = self.reader_gate_cv.wait(gate).map_err(|e| {
                    CoreError::ComputationError(ErrorContext::new(format!(
                        "condvar wait failed: {e}"
                    )))
                })?;
            }
        }
        let guard = self.data.read().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("rwlock poisoned: {e}")))
        })?;
        Ok(ReadGuard { guard })
    }

    /// Acquire a write lock.
    ///
    /// Writers are given priority: once a writer starts waiting,
    /// new readers are blocked until the writer is serviced.
    pub fn write(&self) -> CoreResult<WriteGuard<'_, T>> {
        self.writers_waiting.fetch_add(1, Ordering::Release);
        let guard = self.data.write().map_err(|e| {
            self.writers_waiting.fetch_sub(1, Ordering::Release);
            CoreError::ComputationError(ErrorContext::new(format!("rwlock poisoned: {e}")))
        })?;
        self.writers_waiting.fetch_sub(1, Ordering::Release);
        // Wake blocked readers now that this writer has the lock.
        self.reader_gate_cv.notify_all();
        Ok(WriteGuard { guard })
    }

    /// Try to acquire a read lock without blocking.
    pub fn try_read(&self) -> CoreResult<Option<ReadGuard<'_, T>>> {
        if self.writers_waiting.load(Ordering::Acquire) > 0 {
            return Ok(None);
        }
        match self.data.try_read() {
            Ok(guard) => Ok(Some(ReadGuard { guard })),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(e)) => Err(CoreError::ComputationError(
                ErrorContext::new(format!("rwlock poisoned: {e}")),
            )),
        }
    }

    /// Try to acquire a write lock without blocking.
    pub fn try_write(&self) -> CoreResult<Option<WriteGuard<'_, T>>> {
        match self.data.try_write() {
            Ok(guard) => Ok(Some(WriteGuard { guard })),
            Err(std::sync::TryLockError::WouldBlock) => Ok(None),
            Err(std::sync::TryLockError::Poisoned(e)) => Err(CoreError::ComputationError(
                ErrorContext::new(format!("rwlock poisoned: {e}")),
            )),
        }
    }
}

unsafe impl<T: Send> Send for WriterPreferenceRwLock<T> {}
unsafe impl<T: Send + Sync> Sync for WriterPreferenceRwLock<T> {}

/// RAII read guard for [`WriterPreferenceRwLock`].
pub struct ReadGuard<'a, T> {
    guard: std::sync::RwLockReadGuard<'a, T>,
}

impl<T> std::ops::Deref for ReadGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

/// RAII write guard for [`WriterPreferenceRwLock`].
pub struct WriteGuard<'a, T> {
    guard: std::sync::RwLockWriteGuard<'a, T>,
}

impl<T> std::ops::Deref for WriteGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> std::ops::DerefMut for WriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

// ===========================================================================
// DoubleBuffer
// ===========================================================================

/// A double-buffered exchange for efficient producer/consumer handoff.
///
/// The producer writes to the "back" buffer while the consumer reads from the
/// "front" buffer.  When the producer is done writing, it swaps the buffers
/// (atomically from the consumer's perspective).
pub struct DoubleBuffer<T> {
    buffers: [Mutex<T>; 2],
    /// 0 or 1 – index of the "front" (consumer) buffer.
    front_index: AtomicUsize,
    /// Signals that a new frame is ready.
    new_frame: Condvar,
    new_frame_mutex: Mutex<bool>,
}

impl<T: Clone> DoubleBuffer<T> {
    /// Create a double buffer with two copies of `initial`.
    pub fn new(initial: T) -> Self {
        Self {
            buffers: [Mutex::new(initial.clone()), Mutex::new(initial)],
            front_index: AtomicUsize::new(0),
            new_frame: Condvar::new(),
            new_frame_mutex: Mutex::new(false),
        }
    }

    /// Read a clone of the current front buffer.
    pub fn read_front(&self) -> CoreResult<T> {
        let idx = self.front_index.load(Ordering::Acquire);
        let guard = self.buffers[idx].lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("buffer mutex poisoned: {e}")))
        })?;
        Ok(guard.clone())
    }

    /// Write to the back buffer via a closure, then swap.
    pub fn write_and_swap<F>(&self, f: F) -> CoreResult<()>
    where
        F: FnOnce(&mut T),
    {
        let front = self.front_index.load(Ordering::Acquire);
        let back = 1 - front;
        {
            let mut guard = self.buffers[back].lock().map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!(
                    "buffer mutex poisoned: {e}"
                )))
            })?;
            f(&mut guard);
        }
        // Swap front and back.
        self.front_index.store(back, Ordering::Release);
        // Signal consumers.
        if let Ok(mut flag) = self.new_frame_mutex.lock() {
            *flag = true;
            self.new_frame.notify_all();
        }
        Ok(())
    }

    /// Block until a new frame is available, then read it.
    pub fn wait_and_read(&self, timeout: Duration) -> CoreResult<Option<T>> {
        let mut flag = self.new_frame_mutex.lock().map_err(|e| {
            CoreError::ComputationError(ErrorContext::new(format!("mutex poisoned: {e}")))
        })?;
        if !*flag {
            let (f, timeout_result) = self.new_frame.wait_timeout(flag, timeout).map_err(|e| {
                CoreError::ComputationError(ErrorContext::new(format!("condvar wait failed: {e}")))
            })?;
            flag = f;
            if timeout_result.timed_out() && !*flag {
                return Ok(None);
            }
        }
        *flag = false;
        drop(flag);
        self.read_front().map(Some)
    }

    /// Replace the back buffer entirely, then swap.
    pub fn publish(&self, value: T) -> CoreResult<()> {
        self.write_and_swap(|buf| {
            *buf = value;
        })
    }
}

unsafe impl<T: Send> Send for DoubleBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for DoubleBuffer<T> {}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // ----- ConcurrentHashMap -----

    #[test]
    fn test_hashmap_basic() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::new();
        assert!(map.is_empty());

        map.insert("a".to_string(), 1);
        map.insert("b".to_string(), 2);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"a".to_string()), Some(1));
        assert_eq!(map.get(&"b".to_string()), Some(2));
        assert_eq!(map.get(&"c".to_string()), None);
    }

    #[test]
    fn test_hashmap_remove() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::new();
        map.insert("x".to_string(), 10);
        assert_eq!(map.remove(&"x".to_string()), Some(10));
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn test_hashmap_concurrent() {
        let map = Arc::new(ConcurrentHashMap::<u64, u64>::new());
        let mut handles = Vec::new();

        for t in 0..8 {
            let m = map.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let key = t * 1000 + i;
                    m.insert(key, key * 2);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread join");
        }

        assert_eq!(map.len(), 8000);
        assert_eq!(map.get(&0), Some(0));
        assert_eq!(map.get(&7999), Some(7999 * 2));
    }

    #[test]
    fn test_hashmap_get_or_insert_with() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::new();
        let v = map.get_or_insert_with("key".to_string(), || 42);
        assert_eq!(v, 42);
        let v2 = map.get_or_insert_with("key".to_string(), || 99);
        assert_eq!(v2, 42); // already inserted
    }

    #[test]
    fn test_hashmap_get_and_modify() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::new();
        map.insert("k".to_string(), 10);
        let result = map.get_and_modify(&"k".to_string(), |v| {
            *v += 5;
            *v
        });
        assert_eq!(result, Some(15));
    }

    #[test]
    fn test_hashmap_keys_and_clear() {
        let map: ConcurrentHashMap<u32, u32> = ConcurrentHashMap::new();
        for i in 0..10 {
            map.insert(i, i);
        }
        let keys = map.keys();
        assert_eq!(keys.len(), 10);
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn test_hashmap_contains_key() {
        let map: ConcurrentHashMap<String, i32> = ConcurrentHashMap::new();
        map.insert("hello".into(), 1);
        assert!(map.contains_key(&"hello".to_string()));
        assert!(!map.contains_key(&"world".to_string()));
    }

    // ----- BoundedQueue -----

    #[test]
    fn test_queue_basic() {
        let q: BoundedQueue<i32> = BoundedQueue::new(4);
        assert!(q.is_empty());
        assert_eq!(q.capacity(), 4);

        q.push(1).expect("push 1");
        q.push(2).expect("push 2");
        assert_eq!(q.len(), 2);

        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_queue_full() {
        let q: BoundedQueue<i32> = BoundedQueue::new(2);
        q.push(1).expect("push");
        q.push(2).expect("push");
        assert!(q.push(3).is_err()); // full
    }

    #[test]
    fn test_queue_concurrent() {
        let q = Arc::new(BoundedQueue::<u32>::new(1024));
        let mut handles = Vec::new();

        // Producers
        for t in 0..4 {
            let q2 = q.clone();
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    let val = t * 250 + i;
                    while q2.push(val).is_err() {
                        thread::yield_now();
                    }
                }
            }));
        }

        // Consumer
        let q3 = q.clone();
        let consumer = thread::spawn(move || {
            let mut count = 0u32;
            while count < 1000 {
                if q3.pop().is_some() {
                    count += 1;
                } else {
                    thread::yield_now();
                }
            }
            count
        });

        for h in handles {
            h.join().expect("producer");
        }
        let total = consumer.join().expect("consumer");
        assert_eq!(total, 1000);
    }

    #[test]
    fn test_queue_close() {
        let q: BoundedQueue<i32> = BoundedQueue::new(8);
        q.push(1).expect("push");
        q.close();
        assert!(q.is_closed());
        assert!(q.push(2).is_err()); // closed
        assert_eq!(q.pop(), Some(1)); // can still drain
    }

    // ----- ConcurrentAccumulator -----

    #[test]
    fn test_accumulator_sum() {
        let acc = ConcurrentAccumulator::new(0i64, |a, b| a + b, 4);
        for i in 1..=100 {
            acc.accumulate(i);
        }
        assert_eq!(acc.result(), 5050);
    }

    #[test]
    fn test_accumulator_concurrent() {
        let acc = Arc::new(ConcurrentAccumulator::new(0u64, |a, b| a + b, 8));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let a = acc.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000u64 {
                    a.accumulate(i);
                }
            }));
        }

        for h in handles {
            h.join().expect("thread");
        }

        // Each thread sums 0..999 = 499500; 8 threads = 3996000
        assert_eq!(acc.result(), 8 * 499500);
    }

    #[test]
    fn test_accumulator_reset() {
        let acc = ConcurrentAccumulator::new(0i32, |a, b| a + b, 4);
        acc.accumulate(10);
        acc.accumulate(20);
        assert_eq!(acc.result(), 30);
        acc.reset();
        assert_eq!(acc.result(), 0);
    }

    // ----- AtomicF64Accumulator -----

    #[test]
    fn test_atomic_f64() {
        let acc = AtomicF64Accumulator::new();
        acc.add(1.0);
        acc.add(2.0);
        acc.add(3.0);
        assert!((acc.value() - 6.0).abs() < 1e-10);
        assert_eq!(acc.count(), 3);
    }

    #[test]
    fn test_atomic_f64_concurrent() {
        let acc = Arc::new(AtomicF64Accumulator::new());
        let mut handles = Vec::new();

        for _ in 0..8 {
            let a = acc.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10000 {
                    a.add(1.0);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        assert!((acc.value() - 80000.0).abs() < 1.0);
        assert_eq!(acc.count(), 80000);
    }

    #[test]
    fn test_atomic_f64_reset() {
        let acc = AtomicF64Accumulator::new();
        acc.add(42.0);
        acc.reset();
        assert!((acc.value()).abs() < 1e-15);
        assert_eq!(acc.count(), 0);
    }

    // ----- WriterPreferenceRwLock -----

    #[test]
    fn test_rwlock_basic() {
        let lock = WriterPreferenceRwLock::new(42);
        {
            let r = lock.read().expect("read");
            assert_eq!(*r, 42);
        }
        {
            let mut w = lock.write().expect("write");
            *w = 99;
        }
        {
            let r = lock.read().expect("read");
            assert_eq!(*r, 99);
        }
    }

    #[test]
    fn test_rwlock_concurrent_readers() {
        let lock = Arc::new(WriterPreferenceRwLock::new(vec![1, 2, 3]));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let l = lock.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let r = l.read().expect("read");
                    assert!(!r.is_empty());
                }
            }));
        }

        for h in handles {
            h.join().expect("thread");
        }
    }

    #[test]
    fn test_rwlock_try_read_write() {
        let lock = WriterPreferenceRwLock::new(0);
        let r = lock.try_read().expect("try_read");
        assert!(r.is_some());
        drop(r);

        let w = lock.try_write().expect("try_write");
        assert!(w.is_some());
    }

    // ----- DoubleBuffer -----

    #[test]
    fn test_double_buffer_basic() {
        let db = DoubleBuffer::new(0i32);
        assert_eq!(db.read_front().expect("read"), 0);

        db.publish(42).expect("publish");
        assert_eq!(db.read_front().expect("read"), 42);
    }

    #[test]
    fn test_double_buffer_write_and_swap() {
        let db = DoubleBuffer::new(vec![0u8; 4]);
        db.write_and_swap(|buf| {
            buf[0] = 1;
            buf[1] = 2;
        })
        .expect("write");
        let front = db.read_front().expect("read");
        assert_eq!(front[0], 1);
        assert_eq!(front[1], 2);
    }

    #[test]
    fn test_double_buffer_wait_timeout() {
        let db = Arc::new(DoubleBuffer::new(0));
        let db2 = db.clone();

        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            db2.publish(99).expect("publish");
        });

        let result = db.wait_and_read(Duration::from_secs(2)).expect("wait");
        assert_eq!(result, Some(99));
        producer.join().expect("producer");
    }

    #[test]
    fn test_double_buffer_concurrent() {
        let db = Arc::new(DoubleBuffer::new(0u64));
        let db_w = db.clone();
        let writer_done = Arc::new(AtomicBool::new(false));
        let writer_done_w = writer_done.clone();

        let writer = thread::spawn(move || {
            for i in 1..=100u64 {
                db_w.publish(i).expect("publish");
            }
            writer_done_w.store(true, Ordering::Release);
        });

        // Reader keeps reading until the writer signals completion, then does a
        // final sweep to capture the last published value. This avoids a flaky
        // fixed iteration count: under heavy system load the writer thread can
        // be slow to schedule, and a fixed-count reader may finish before any
        // publish is observable.
        let db_r = db.clone();
        let writer_done_r = writer_done.clone();
        let reader = thread::spawn(move || {
            let mut max_seen = 0u64;
            loop {
                let v = db_r.read_front().expect("read");
                if v > max_seen {
                    max_seen = v;
                }
                if writer_done_r.load(Ordering::Acquire) {
                    // Final reads to catch the last published value.
                    for _ in 0..16 {
                        let v = db_r.read_front().expect("read");
                        if v > max_seen {
                            max_seen = v;
                        }
                    }
                    break;
                }
                thread::yield_now();
            }
            max_seen
        });

        writer.join().expect("writer");
        let max_seen = reader.join().expect("reader");
        assert!(max_seen > 0);
        // Writer published values up to 100; reader must have caught the final
        // one after the writer-done signal.
        assert_eq!(max_seen, 100);
    }
}

// ---------------------------------------------------------------------------
// Lock-free data structure submodules
// ---------------------------------------------------------------------------

pub mod compressed_trie;
pub mod persistent_vector;
pub mod queue;
pub mod skip_list;
pub mod stack;

pub use compressed_trie::CompressedTrie;
pub use persistent_vector::PersistentRrbVec;
pub use queue::LockFreeQueue;
pub use skip_list::SkipList;
pub use stack::LockFreeStack;

// ---------------------------------------------------------------------------
// Advanced concurrency submodules
// ---------------------------------------------------------------------------

pub mod async_utils;
pub mod barrier;
pub mod parallel_iter;
pub mod work_stealing;
pub use async_utils as concurrent_async;

pub use barrier::{CountDownLatch, CyclicBarrier, PhaseBarrier, SpinBarrier};
pub use concurrent_async::{
    BackoffStrategy, FutureExecutor, JoinFuture, RetryPolicy, Semaphore, SemaphoreGuard,
    TokenBucketRateLimiter,
};
pub use parallel_iter::{
    parallel_filter, parallel_for_each, parallel_map, parallel_merge_sort, parallel_partition,
    parallel_prefix_sum, parallel_reduce, parallel_scan, ScanMode,
};
pub use work_stealing::{
    Priority, PriorityTaskQueue, SchedulerConfig, SchedulerStats, StealResult, WorkStealingDeque,
    WorkStealingScheduler,
};
