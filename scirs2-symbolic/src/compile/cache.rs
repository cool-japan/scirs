//! Hash-keyed JIT compilation cache.
//!
//! [`JitCache`] maps a `LoweredOp::structural_hash()` (`u128`) to an
//! `Arc<JitFunction>`. Lookup is `O(1)`; on miss the caller compiles
//! and inserts. A naive eviction policy is used: when the cache reaches
//! capacity, an arbitrary entry is removed before insertion. This is
//! adequate for typical SR / regression workloads where the same handful
//! of formulas are evaluated repeatedly; full LRU is a v0.4.5 follow-up.

#![cfg(feature = "jit")]

use crate::compile::jit::JitFunction;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Default capacity used by [`JitCache::default`].
pub const DEFAULT_JIT_CACHE_CAPACITY: usize = 256;

/// Hash-keyed cache of JIT-compiled `LoweredOp` formulas.
///
/// Thread-safe: cloning is cheap (`Arc`-shared) and look-ups acquire
/// a `Mutex`. On a poisoned lock the cache appears empty but never
/// panics — see the no-`unwrap()` policy.
#[derive(Clone)]
pub struct JitCache {
    inner: Arc<Mutex<HashMap<u128, Arc<JitFunction>>>>,
    capacity: usize,
}

impl JitCache {
    /// Create a new cache with the given maximum entry count.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            capacity: capacity.max(1),
        }
    }

    /// Look up a compiled function by its structural hash.
    ///
    /// Returns `None` on miss or if the inner mutex is poisoned.
    pub fn get(&self, hash: u128) -> Option<Arc<JitFunction>> {
        let guard = self.inner.lock().ok()?;
        guard.get(&hash).cloned()
    }

    /// Insert a compiled function under the given hash.
    ///
    /// If the cache is at capacity, an arbitrary existing entry is
    /// evicted to make room. A poisoned lock is silently ignored
    /// (the function is simply not cached).
    pub fn insert(&self, hash: u128, func: Arc<JitFunction>) {
        if let Ok(mut guard) = self.inner.lock() {
            if !guard.contains_key(&hash) && guard.len() >= self.capacity {
                let victim = guard.keys().next().copied();
                if let Some(k) = victim {
                    guard.remove(&k);
                }
            }
            guard.insert(hash, func);
        }
    }

    /// Number of cached compiled functions.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// True when the cache holds zero entries (or the lock is poisoned).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache capacity (max entries before eviction).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all cached functions.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.clear();
        }
    }
}

impl Default for JitCache {
    fn default() -> Self {
        Self::new(DEFAULT_JIT_CACHE_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::jit::to_jit;
    use crate::eml::op::LoweredOp;

    #[test]
    fn cache_default_capacity() {
        let cache = JitCache::default();
        assert_eq!(cache.capacity(), DEFAULT_JIT_CACHE_CAPACITY);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_insert_and_get_roundtrip() {
        let cache = JitCache::new(8);
        let op = LoweredOp::Const(2.5);
        let hash = op.structural_hash();
        let func = Arc::new(to_jit(&op).expect("compile const"));
        cache.insert(hash, Arc::clone(&func));
        assert_eq!(cache.len(), 1);
        let got = cache.get(hash).expect("cache hit");
        assert_eq!(got.eval(&[]), 2.5);
    }

    #[test]
    fn cache_miss_returns_none() {
        let cache = JitCache::new(4);
        let bogus_hash = 0xdead_beef_cafe_u128;
        assert!(cache.get(bogus_hash).is_none());
    }

    #[test]
    fn cache_eviction_at_capacity() {
        let cache = JitCache::new(2);
        for i in 0..5 {
            let op = LoweredOp::Const(i as f64);
            let hash = op.structural_hash();
            let func = Arc::new(to_jit(&op).expect("compile"));
            cache.insert(hash, func);
        }
        // Capacity is 2, inserted 5 — should never grow past capacity.
        assert!(cache.len() <= 2);
    }

    #[test]
    fn cache_clear_empties() {
        let cache = JitCache::new(4);
        let op = LoweredOp::Const(1.0);
        cache.insert(
            op.structural_hash(),
            Arc::new(to_jit(&op).expect("compile")),
        );
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cache_is_clone_for_sharing() {
        let cache = JitCache::new(4);
        let op = LoweredOp::Const(7.0);
        let hash = op.structural_hash();
        cache.insert(hash, Arc::new(to_jit(&op).expect("compile")));
        let cloned = cache.clone();
        assert_eq!(cloned.len(), 1);
        let f = cloned.get(hash).expect("hit through clone");
        assert_eq!(f.eval(&[]), 7.0);
    }
}
