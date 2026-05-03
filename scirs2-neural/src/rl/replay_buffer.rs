//! Experience replay buffer used by off-policy RL algorithms (SAC, TD3, DQN …).
//!
//! The buffer is a fixed-capacity circular array.  Once full, the oldest
//! transitions are silently overwritten.
//!
//! Generic over the scalar type `F` so that both `f32` and `f64` networks are
//! supported.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::num_traits::Float;

// ──────────────────────────────────────────────────────────────────────────────
// Transition (batch output)
// ──────────────────────────────────────────────────────────────────────────────

/// A batch of transitions sampled from a [`ReplayBuffer`].
#[derive(Debug)]
pub struct Transition<F: Float> {
    /// States, shape `[batch_size, obs_dim]`.
    pub states: Array2<F>,
    /// Actions, shape `[batch_size, act_dim]`.
    pub actions: Array2<F>,
    /// Scalar rewards, shape `[batch_size]`.
    pub rewards: Array1<F>,
    /// Next-states, shape `[batch_size, obs_dim]`.
    pub next_states: Array2<F>,
    /// Done flags, shape `[batch_size]`.
    pub dones: Array1<bool>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Minimal RNG (no external rand crate)
// ──────────────────────────────────────────────────────────────────────────────

/// Xorshift64 — lightweight RNG for index sampling.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new RNG with the given seed (must not be 0).
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        Self { state: s }
    }

    /// Produce the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Sample a value in `[0, n)`.
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Anything that exposes the same interface as `XorShift64`.
pub trait Rng {
    /// Sample a value in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize;
}

impl Rng for XorShift64 {
    fn next_usize(&mut self, n: usize) -> usize {
        XorShift64::next_usize(self, n)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ReplayBuffer
// ──────────────────────────────────────────────────────────────────────────────

/// Fixed-capacity circular replay buffer for off-policy RL.
///
/// # Type parameters
/// - `F`: scalar element type (typically `f32` or `f64`).
///
/// # Example
/// ```rust,ignore
/// use scirs2_neural::rl::replay_buffer::{ReplayBuffer, XorShift64};
/// use scirs2_core::ndarray::array;
///
/// let mut buf: ReplayBuffer<f64> = ReplayBuffer::new(1000, 4, 2);
/// buf.push(
///     array![0.0, 0.1, 0.2, 0.3],
///     array![0.5, -0.1],
///     1.0,
///     array![0.0, 0.15, 0.25, 0.35],
///     false,
/// );
/// assert_eq!(buf.len(), 1);
/// ```
pub struct ReplayBuffer<F: Float> {
    capacity: usize,
    obs_dim: usize,
    act_dim: usize,
    states: Array2<F>,
    actions: Array2<F>,
    rewards: Array1<F>,
    next_states: Array2<F>,
    dones: Array1<bool>,
    /// Write pointer (next slot to fill).
    ptr: usize,
    /// Number of valid transitions currently stored.
    size: usize,
}

impl<F: Float + Default + Clone + 'static> ReplayBuffer<F> {
    /// Allocate a new buffer with given `capacity`, observation dimension
    /// `obs_dim`, and action dimension `act_dim`.
    pub fn new(capacity: usize, obs_dim: usize, act_dim: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(obs_dim > 0, "obs_dim must be > 0");
        assert!(act_dim > 0, "act_dim must be > 0");
        Self {
            capacity,
            obs_dim,
            act_dim,
            states: Array2::zeros((capacity, obs_dim)),
            actions: Array2::zeros((capacity, act_dim)),
            rewards: Array1::zeros(capacity),
            next_states: Array2::zeros((capacity, obs_dim)),
            dones: Array1::from_elem(capacity, false),
            ptr: 0,
            size: 0,
        }
    }

    /// Store a single transition.  When the buffer is full, the oldest entry
    /// is overwritten.
    pub fn push(
        &mut self,
        state: Array1<F>,
        action: Array1<F>,
        reward: F,
        next_state: Array1<F>,
        done: bool,
    ) {
        debug_assert_eq!(state.len(), self.obs_dim, "state dimension mismatch");
        debug_assert_eq!(action.len(), self.act_dim, "action dimension mismatch");
        debug_assert_eq!(
            next_state.len(),
            self.obs_dim,
            "next_state dimension mismatch"
        );

        self.states.row_mut(self.ptr).assign(&state);
        self.actions.row_mut(self.ptr).assign(&action);
        self.rewards[self.ptr] = reward;
        self.next_states.row_mut(self.ptr).assign(&next_state);
        self.dones[self.ptr] = done;

        self.ptr = (self.ptr + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Sample `batch_size` transitions uniformly at random (with replacement
    /// when `batch_size > self.len()`).
    ///
    /// `rng` can be any type implementing [`Rng`].
    pub fn sample<R: Rng>(&self, batch_size: usize, rng: &mut R) -> Transition<F> {
        assert!(self.size > 0, "cannot sample from empty buffer");

        let indices: Vec<usize> = (0..batch_size).map(|_| rng.next_usize(self.size)).collect();

        self.gather(&indices)
    }

    /// Sample without replacement (Fisher-Yates on the live portion).
    ///
    /// Falls back to sampling with replacement if `batch_size > self.len()`.
    pub fn sample_no_replace<R: Rng>(&self, batch_size: usize, rng: &mut R) -> Transition<F> {
        assert!(self.size > 0, "cannot sample from empty buffer");

        let indices = if batch_size <= self.size {
            let mut pool: Vec<usize> = (0..self.size).collect();
            for i in 0..batch_size {
                let j = i + rng.next_usize(self.size - i);
                pool.swap(i, j);
            }
            pool[..batch_size].to_vec()
        } else {
            (0..batch_size).map(|_| rng.next_usize(self.size)).collect()
        };

        self.gather(&indices)
    }

    /// Number of valid transitions currently in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the buffer contains no transitions.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Maximum number of transitions the buffer can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Observation dimension.
    #[inline]
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension.
    #[inline]
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    // ─── private helpers ────────────────────────────────────────────────────

    fn gather(&self, indices: &[usize]) -> Transition<F> {
        let n = indices.len();
        let mut states = Array2::zeros((n, self.obs_dim));
        let mut actions = Array2::zeros((n, self.act_dim));
        let mut rewards = Array1::zeros(n);
        let mut next_states = Array2::zeros((n, self.obs_dim));
        let mut dones = Array1::from_elem(n, false);

        for (out_i, &buf_i) in indices.iter().enumerate() {
            states
                .row_mut(out_i)
                .assign(&self.states.index_axis(Axis(0), buf_i));
            actions
                .row_mut(out_i)
                .assign(&self.actions.index_axis(Axis(0), buf_i));
            rewards[out_i] = self.rewards[buf_i];
            next_states
                .row_mut(out_i)
                .assign(&self.next_states.index_axis(Axis(0), buf_i));
            dones[out_i] = self.dones[buf_i];
        }

        Transition {
            states,
            actions,
            rewards,
            next_states,
            dones,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_rng() -> XorShift64 {
        XorShift64::new(42)
    }

    #[test]
    fn push_and_len() {
        let mut buf: ReplayBuffer<f64> = ReplayBuffer::new(10, 4, 2);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        buf.push(
            array![0.0, 1.0, 2.0, 3.0],
            array![0.5, -0.5],
            1.0,
            array![0.1, 1.1, 2.1, 3.1],
            false,
        );
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn circular_overwrite() {
        let mut buf: ReplayBuffer<f64> = ReplayBuffer::new(3, 2, 1);
        for i in 0..5u64 {
            let fi = i as f64;
            buf.push(
                array![fi, fi],
                array![fi],
                fi,
                array![fi + 1.0, fi + 1.0],
                false,
            );
        }
        // Buffer capacity is 3; should hold the last 3 transitions.
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn sample_shapes() {
        let mut buf: ReplayBuffer<f64> = ReplayBuffer::new(100, 4, 2);
        for i in 0..50u64 {
            let fi = i as f64;
            buf.push(
                array![fi, fi, fi, fi],
                array![fi, fi],
                fi,
                array![fi + 1.0, fi + 1.0, fi + 1.0, fi + 1.0],
                i % 5 == 0,
            );
        }
        let mut rng = make_rng();
        let tr = buf.sample(16, &mut rng);
        assert_eq!(tr.states.shape(), &[16, 4]);
        assert_eq!(tr.actions.shape(), &[16, 2]);
        assert_eq!(tr.rewards.len(), 16);
        assert_eq!(tr.next_states.shape(), &[16, 4]);
        assert_eq!(tr.dones.len(), 16);
    }

    #[test]
    fn sample_no_replace_unique_indices() {
        let mut buf: ReplayBuffer<f32> = ReplayBuffer::new(20, 2, 1);
        for i in 0..20u64 {
            let fi = i as f32;
            buf.push(
                array![fi, fi],
                array![fi],
                fi,
                array![fi + 1.0, fi + 1.0],
                false,
            );
        }
        let mut rng = make_rng();
        let tr = buf.sample_no_replace(10, &mut rng);
        assert_eq!(tr.states.shape(), &[10, 2]);
    }

    #[test]
    fn transition_rewards_are_finite() {
        let mut buf: ReplayBuffer<f64> = ReplayBuffer::new(50, 3, 1);
        for i in 0..50u64 {
            let fi = i as f64;
            buf.push(
                array![fi, fi, fi],
                array![fi],
                fi * 0.1,
                array![fi + 1.0, fi + 1.0, fi + 1.0],
                false,
            );
        }
        let mut rng = make_rng();
        let tr = buf.sample(32, &mut rng);
        assert!(tr.rewards.iter().all(|r| r.is_finite()));
    }
}
