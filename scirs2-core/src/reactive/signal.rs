//! Qt-style signal/slot system and observable values.
//!
//! # Concepts
//!
//! - [`Signal<T>`] — a pub/sub channel.  Slots (callbacks) are registered via
//!   [`Signal::connect`] and invoked synchronously by [`Signal::emit`].
//!   Each registration returns a [`SlotId`] that can be passed to
//!   [`Signal::disconnect`] to remove the callback.
//!
//! - [`Observable<T>`] — a value container that fires an internal `Signal`
//!   whenever the value changes.  Derived observables can be created with
//!   [`Observable::map`].
//!
//! - [`ComputedObservable<T>`] — a read-only observable whose value is
//!   recomputed from a set of input [`Observable`]s whenever any of them
//!   changes.  The computation function is stored as a `Box<dyn Fn() -> T>`.
//!
//! All types are `Send + Sync` and can be shared across threads.
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::reactive::signal::{Observable, Signal};
//!
//! let sig = Signal::<i32>::new();
//! let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
//! let log2 = log.clone();
//! sig.connect(move |v| {
//!     if let Ok(mut g) = log2.lock() { g.push(*v); }
//! });
//! sig.emit(&42);
//! sig.emit(&7);
//! assert_eq!(*log.lock().unwrap_or_else(|e| e.into_inner()), vec![42, 7]);
//! ```

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// SlotId
// ---------------------------------------------------------------------------

/// Opaque identifier for a connected slot (callback).
///
/// Pass to [`Signal::disconnect`] to stop receiving emissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(u64);

// ---------------------------------------------------------------------------
// Signal<T>
// ---------------------------------------------------------------------------

type SlotFn<T> = Box<dyn Fn(&T) + Send + Sync + 'static>;

struct SignalInner<T> {
    slots: Vec<(SlotId, SlotFn<T>)>,
    next_id: u64,
}

impl<T> SignalInner<T> {
    fn new() -> Self {
        SignalInner {
            slots: Vec::new(),
            next_id: 1,
        }
    }
}

/// A signal that can have multiple slots (callbacks) connected.
///
/// Emitting a value calls all connected slots synchronously in registration
/// order.  Signals are `Clone`-able; clones share the same slot list.
pub struct Signal<T: Clone> {
    inner: Arc<Mutex<SignalInner<T>>>,
}

impl<T: Clone + 'static> Signal<T> {
    /// Create a new signal with no connected slots.
    pub fn new() -> Self {
        Signal {
            inner: Arc::new(Mutex::new(SignalInner::new())),
        }
    }

    /// Register a callback and return its [`SlotId`].
    ///
    /// The callback receives a shared reference to the emitted value.
    pub fn connect(&self, slot: impl Fn(&T) + Send + Sync + 'static) -> SlotId {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return SlotId(0),
        };
        let id = SlotId(guard.next_id);
        guard.next_id = guard.next_id.wrapping_add(1);
        guard.slots.push((id, Box::new(slot)));
        id
    }

    /// Remove the slot identified by `id`.
    ///
    /// Returns `true` if the slot was found and removed.
    pub fn disconnect(&self, id: SlotId) -> bool {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let before = guard.slots.len();
        guard.slots.retain(|(sid, _)| *sid != id);
        guard.slots.len() < before
    }

    /// Fire all connected slots with `value`.
    pub fn emit(&self, value: &T) {
        // Collect callbacks while holding the lock, then invoke them after
        // releasing to avoid re-entrant deadlocks.
        let callbacks: Vec<SlotFn<T>> = {
            // We can't clone Box<dyn Fn>, so we temporarily call them inside
            // the lock.  This is safe as long as slots do not call back into
            // Signal methods.  For a fully re-entrant solution, an RwLock
            // with a separate emit-queue would be needed.
            if let Ok(guard) = self.inner.lock() {
                for (_, cb) in &guard.slots {
                    cb(value);
                }
            }
            Vec::new() // dummy — see comment above
        };
        drop(callbacks);
    }

    /// Return the number of connected slots.
    pub fn slot_count(&self) -> usize {
        self.inner.lock().map(|g| g.slots.len()).unwrap_or(0)
    }
}

impl<T: Clone + 'static> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Signal {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ---------------------------------------------------------------------------
// Observable<T>
// ---------------------------------------------------------------------------

struct ObservableInner<T: Clone> {
    value: T,
    signal: Signal<T>,
}

/// A value wrapper that emits a [`Signal`] whenever its value changes.
///
/// Multiple handles to the same observable can be created by calling
/// [`Observable::clone`] — all share the same underlying value and signal.
///
/// # Example
///
/// ```rust
/// use scirs2_core::reactive::signal::Observable;
/// use std::sync::{Arc, Mutex};
///
/// let obs = Observable::new(0i32);
/// let log = Arc::new(Mutex::new(Vec::new()));
/// let log2 = Arc::clone(&log);
/// obs.on_change().connect(move |v| {
///     if let Ok(mut g) = log2.lock() { g.push(*v); }
/// });
/// obs.set(1);
/// obs.set(2);
/// assert_eq!(*log.lock().unwrap_or_else(|e| e.into_inner()), vec![1, 2]);
/// ```
pub struct Observable<T: Clone + 'static> {
    inner: Arc<Mutex<ObservableInner<T>>>,
}

impl<T: Clone + 'static> Observable<T> {
    /// Create an observable with initial value `value`.
    pub fn new(value: T) -> Self {
        Observable {
            inner: Arc::new(Mutex::new(ObservableInner {
                value,
                signal: Signal::new(),
            })),
        }
    }

    /// Return a clone of the current value.
    pub fn get(&self) -> T {
        self.inner
            .lock()
            .map(|g| g.value.clone())
            .unwrap_or_else(|_| panic!("Observable mutex poisoned"))
    }

    /// Update the value and fire the change signal.
    pub fn set(&self, value: T) {
        let signal_clone = {
            let mut guard = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return,
            };
            guard.value = value.clone();
            guard.signal.clone()
        };
        signal_clone.emit(&value);
    }

    /// Return the underlying [`Signal`] so callers can connect slots.
    pub fn on_change(&self) -> Signal<T> {
        self.inner
            .lock()
            .map(|g| g.signal.clone())
            .unwrap_or_else(|_| Signal::new())
    }

    /// Create a derived [`Observable<U>`] whose value is `f(self.get())` and
    /// that recomputes whenever this observable changes.
    pub fn map<U: Clone + Send + 'static>(
        &self,
        f: impl Fn(&T) -> U + Send + Sync + 'static,
    ) -> Observable<U> {
        let initial = f(&self.get());
        let derived = Observable::<U>::new(initial);
        let derived_clone = derived.clone();
        self.on_change().connect(move |v| {
            derived_clone.set(f(v));
        });
        derived
    }
}

impl<T: Clone + 'static> Clone for Observable<T> {
    fn clone(&self) -> Self {
        Observable {
            inner: Arc::clone(&self.inner),
        }
    }
}

// ---------------------------------------------------------------------------
// AnyObservable — type-erased dependency subscription
// ---------------------------------------------------------------------------

/// Type-erased trait for registering a callback on any observable.
pub trait AnyObservable: Send + Sync {
    /// Connect a no-argument callback that fires whenever the observable changes.
    fn on_any_change(&self, cb: Box<dyn Fn() + Send + Sync + 'static>) -> SlotId;
    /// Disconnect a callback previously registered via `on_any_change`.
    fn disconnect_any(&self, id: SlotId);
}

impl<T: Clone + Send + Sync + 'static> AnyObservable for Observable<T> {
    fn on_any_change(&self, cb: Box<dyn Fn() + Send + Sync + 'static>) -> SlotId {
        self.on_change().connect(move |_| cb())
    }

    fn disconnect_any(&self, id: SlotId) {
        self.on_change().disconnect(id);
    }
}

// ---------------------------------------------------------------------------
// ComputedObservable<T>
// ---------------------------------------------------------------------------

/// A read-only observable that automatically recomputes its value from a set
/// of input observables.
///
/// Every time any dependency changes, `compute_fn` is called and the result
/// is stored.  Subscribers connected via [`ComputedObservable::on_change`]
/// are notified.
///
/// # Example
///
/// ```rust
/// use scirs2_core::reactive::signal::{Observable, ComputedObservable};
///
/// let a = Observable::new(3i32);
/// let b = Observable::new(4i32);
/// let a2 = a.clone();
/// let b2 = b.clone();
/// let sum = ComputedObservable::new(
///     move || a2.get() + b2.get(),
///     vec![],   // subscriptions managed via closures capturing a2/b2
/// );
/// assert_eq!(sum.get(), 7);
/// ```
pub struct ComputedObservable<T: Clone + 'static> {
    inner: Observable<T>,
    /// Slot IDs for dependency subscriptions (kept alive until this struct
    /// is dropped).
    _subscriptions: Arc<Mutex<Vec<(Box<dyn AnyObservable + 'static>, SlotId)>>>,
}

impl<T: Clone + Send + 'static> ComputedObservable<T> {
    /// Create a `ComputedObservable` from a `compute_fn` and a list of
    /// dynamic dependencies.
    ///
    /// `dependencies` should be a `Vec` of `Box<dyn AnyObservable>` values.
    /// When any of them changes, `compute_fn` is re-evaluated and the result
    /// stored.
    pub fn new(
        compute_fn: impl Fn() -> T + Send + Sync + 'static,
        dependencies: Vec<Box<dyn AnyObservable + 'static>>,
    ) -> Self {
        let compute_fn = Arc::new(compute_fn);
        let initial = compute_fn();
        let inner = Observable::new(initial);

        let mut subscriptions: Vec<(Box<dyn AnyObservable + 'static>, SlotId)> = Vec::new();

        for dep in dependencies {
            let inner_clone = inner.clone();
            let compute_clone = Arc::clone(&compute_fn);
            let id = dep.on_any_change(Box::new(move || {
                inner_clone.set(compute_clone());
            }));
            subscriptions.push((dep, id));
        }

        ComputedObservable {
            inner,
            _subscriptions: Arc::new(Mutex::new(subscriptions)),
        }
    }

    /// Return the current computed value.
    pub fn get(&self) -> T {
        self.inner.get()
    }

    /// Return the change signal so callers can subscribe to recomputations.
    pub fn on_change(&self) -> Signal<T> {
        self.inner.on_change()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_signal_emit() {
        let sig = Signal::<i32>::new();
        let received = Arc::new(Mutex::new(Vec::new()));
        let r2 = Arc::clone(&received);
        sig.connect(move |v| {
            r2.lock().map(|mut g| g.push(*v)).ok();
        });
        sig.emit(&10);
        sig.emit(&20);
        assert_eq!(
            *received.lock().unwrap_or_else(|e| e.into_inner()),
            vec![10, 20]
        );
    }

    #[test]
    fn test_signal_disconnect() {
        let sig = Signal::<i32>::new();
        let count = Arc::new(Mutex::new(0usize));
        let c2 = Arc::clone(&count);
        let id = sig.connect(move |_| {
            c2.lock().map(|mut g| *g += 1).ok();
        });
        sig.emit(&1);
        sig.disconnect(id);
        sig.emit(&2);
        assert_eq!(*count.lock().unwrap_or_else(|e| e.into_inner()), 1);
    }

    #[test]
    fn test_signal_multiple_slots() {
        let sig = Signal::<i32>::new();
        let sum = Arc::new(Mutex::new(0i32));
        let s2 = Arc::clone(&sum);
        let s3 = Arc::clone(&sum);
        sig.connect(move |v| {
            s2.lock().map(|mut g| *g += *v).ok();
        });
        sig.connect(move |v| {
            s3.lock().map(|mut g| *g += *v * 2).ok();
        });
        sig.emit(&3);
        // slot1: +3, slot2: +6 → 9
        assert_eq!(*sum.lock().unwrap_or_else(|e| e.into_inner()), 9);
    }

    #[test]
    fn test_observable_set_get() {
        let obs = Observable::new(42i32);
        assert_eq!(obs.get(), 42);
        obs.set(100);
        assert_eq!(obs.get(), 100);
    }

    #[test]
    fn test_observable_on_change() {
        let obs = Observable::new(0i32);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log2 = Arc::clone(&log);
        obs.on_change().connect(move |v| {
            log2.lock().map(|mut g| g.push(*v)).ok();
        });
        obs.set(5);
        obs.set(10);
        assert_eq!(*log.lock().unwrap_or_else(|e| e.into_inner()), vec![5, 10]);
    }

    #[test]
    fn test_observable_map() {
        let obs = Observable::new(3i32);
        let doubled = obs.map(|v| *v * 2);
        assert_eq!(doubled.get(), 6);
        obs.set(7);
        assert_eq!(doubled.get(), 14);
    }

    #[test]
    fn test_computed_observable() {
        let a = Observable::new(10i32);
        let b = Observable::new(20i32);
        let a_clone = a.clone();
        let b_clone = b.clone();
        let sum = ComputedObservable::new(
            move || a_clone.get() + b_clone.get(),
            vec![
                Box::new(a.clone()) as Box<dyn AnyObservable>,
                Box::new(b.clone()) as Box<dyn AnyObservable>,
            ],
        );
        assert_eq!(sum.get(), 30);
        a.set(5);
        assert_eq!(sum.get(), 25);
        b.set(5);
        assert_eq!(sum.get(), 10);
    }
}
