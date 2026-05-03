//! Experience replay buffers for reinforcement learning

use crate::error::{NeuralError, Result};
use crate::reinforcement::ExperienceBatch;
use scirs2_core::ndarray::prelude::*;
use std::collections::VecDeque;

/// Trait for experience replay buffers
pub trait ReplayBufferTrait: Send + Sync {
    /// Add an experience to the buffer
    fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()>;

    /// Sample a random batch of experiences
    fn sample_batch(&self, batch_size: usize) -> Result<ExperienceBatch>;

    /// Current number of stored experiences
    fn len(&self) -> usize;

    /// Whether the buffer is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maximum capacity
    fn capacity(&self) -> usize;

    /// Persist buffer to disk
    fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    /// Load buffer from disk
    fn load(&mut self, _path: &str) -> Result<()> {
        Ok(())
    }
}

/// A single stored experience
#[derive(Clone, Debug)]
pub struct Experience {
    pub state: Array1<f32>,
    pub action: Array1<f32>,
    pub reward: f32,
    pub next_state: Array1<f32>,
    pub done: bool,
    pub info: Option<std::collections::HashMap<String, f32>>,
}

/// Type alias kept for API compatibility
pub type SimpleReplayBuffer = ReplayBuffer;

/// Standard uniform replay buffer
pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
    /// Simple xorshift64 state for sampling without `rand`
    rng_state: u64,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            rng_state: 0xdeadbeef_cafebabe,
        }
    }

    /// Add a raw experience
    pub fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        let exp = Experience {
            state,
            action,
            reward,
            next_state,
            done,
            info: None,
        };
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
        Ok(())
    }

    /// Add an experience with extra metadata
    pub fn add_with_info(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
        info: std::collections::HashMap<String, f32>,
    ) -> Result<()> {
        let exp = Experience {
            state,
            action,
            reward,
            next_state,
            done,
            info: Some(info),
        };
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(exp);
        Ok(())
    }

    /// Sample `batch_size` experiences uniformly at random
    pub fn sample(&mut self, batch_size: usize) -> Result<ExperienceBatch> {
        if self.buffer.len() < batch_size {
            return Err(NeuralError::InvalidArgument(format!(
                "Not enough experiences: {} < {}",
                self.buffer.len(),
                batch_size
            )));
        }
        let n = self.buffer.len();
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle (first batch_size elements)
        for i in 0..batch_size {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let j = i + (self.rng_state as usize % (n - i));
            indices.swap(i, j);
        }
        let chosen: Vec<&Experience> = indices[..batch_size]
            .iter()
            .map(|&i| &self.buffer[i])
            .collect();

        let state_dim = chosen[0].state.len();
        let action_dim = chosen[0].action.len();

        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Array2::zeros((batch_size, action_dim));
        let mut rewards = Array1::zeros(batch_size);
        let mut next_states = Array2::zeros((batch_size, state_dim));
        let mut dones = Array1::from_elem(batch_size, false);

        for (i, exp) in chosen.iter().enumerate() {
            states.row_mut(i).assign(&exp.state);
            actions.row_mut(i).assign(&exp.action);
            rewards[i] = exp.reward;
            next_states.row_mut(i).assign(&exp.next_state);
            dones[i] = exp.done;
        }
        Ok(ExperienceBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
            info: None,
        })
    }

    /// Number of stored experiences
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl ReplayBufferTrait for ReplayBuffer {
    fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        ReplayBuffer::add(self, state, action, reward, next_state, done)
    }

    fn sample_batch(&self, _batch_size: usize) -> Result<ExperienceBatch> {
        // Note: `sample` requires &mut self for the internal RNG. For the
        // trait method (shared reference), return an error asking the caller
        // to use the concrete `sample()` method instead.
        Err(NeuralError::InvalidArgument(
            "Use ReplayBuffer::sample(&mut self, batch_size) for mutable sampling".to_string(),
        ))
    }

    fn len(&self) -> usize {
        ReplayBuffer::len(self)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Prioritized Replay Buffer ─────────────────────────────────────────────────

/// Experience with a priority weight for prioritized replay
#[derive(Clone, Debug)]
struct PrioritizedExperience {
    exp: Experience,
    priority: f32,
}

/// Prioritized experience replay buffer (PER, Schaul et al. 2016)
pub struct PrioritizedReplayBuffer {
    buffer: Vec<PrioritizedExperience>,
    capacity: usize,
    alpha: f32,
    beta: f32,
    max_priority: f32,
    rng_state: u64,
}

impl PrioritizedReplayBuffer {
    /// Create a new prioritized replay buffer
    pub fn new(capacity: usize, alpha: f32, beta0: f32) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            alpha,
            beta: beta0,
            max_priority: 1.0,
            rng_state: 0xcafebabe_deadbeef,
        }
    }

    /// Add experience with current max priority
    pub fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        let exp = Experience {
            state,
            action,
            reward,
            next_state,
            done,
            info: None,
        };
        let prio = PrioritizedExperience {
            exp,
            priority: self.max_priority,
        };
        if self.buffer.len() >= self.capacity {
            // Evict the experience with the lowest priority
            if let Some(min_idx) = self
                .buffer
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.priority.partial_cmp(&b.1.priority).expect("non-NaN"))
                .map(|(i, _)| i)
            {
                self.buffer.remove(min_idx);
            }
        }
        self.buffer.push(prio);
        Ok(())
    }

    /// Sample `batch_size` experiences by priority
    ///
    /// Returns `(batch, importance_weights, sampled_indices)`
    pub fn sample(
        &mut self,
        batch_size: usize,
    ) -> Result<(ExperienceBatch, Array1<f32>, Vec<usize>)> {
        if self.buffer.len() < batch_size {
            return Err(NeuralError::InvalidArgument(format!(
                "Not enough prioritized experiences: {} < {}",
                self.buffer.len(),
                batch_size
            )));
        }
        // Compute sampling probabilities
        let priorities: Vec<f32> = self
            .buffer
            .iter()
            .map(|e| e.priority.powf(self.alpha))
            .collect();
        let total: f32 = priorities.iter().sum();
        let probs: Vec<f32> = priorities.iter().map(|p| p / total.max(1e-10)).collect();
        let n = self.buffer.len();

        // Stratified sampling
        let mut chosen_indices = Vec::with_capacity(batch_size);
        for seg in 0..batch_size {
            let lo = seg as f32 / batch_size as f32;
            let hi = (seg + 1) as f32 / batch_size as f32;
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let u = lo + ((self.rng_state as f32 / u64::MAX as f32) * (hi - lo));
            let mut cumsum = 0.0f32;
            let mut selected = n - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= u {
                    selected = i;
                    break;
                }
            }
            chosen_indices.push(selected);
        }

        // Compute importance-sampling weights
        let min_prob = probs
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
            .max(1e-10);
        let max_weight = (min_prob * n as f32).powf(-self.beta);
        let weights: Array1<f32> = Array1::from_vec(
            chosen_indices
                .iter()
                .map(|&i| {
                    let w = (probs[i] * n as f32).powf(-self.beta);
                    w / max_weight
                })
                .collect(),
        );

        let chosen: Vec<&Experience> = chosen_indices
            .iter()
            .map(|&i| &self.buffer[i].exp)
            .collect();
        let state_dim = chosen[0].state.len();
        let action_dim = chosen[0].action.len();

        let mut states = Array2::zeros((batch_size, state_dim));
        let mut actions = Array2::zeros((batch_size, action_dim));
        let mut rewards = Array1::zeros(batch_size);
        let mut next_states = Array2::zeros((batch_size, state_dim));
        let mut dones = Array1::from_elem(batch_size, false);

        for (i, exp) in chosen.iter().enumerate() {
            states.row_mut(i).assign(&exp.state);
            actions.row_mut(i).assign(&exp.action);
            rewards[i] = exp.reward;
            next_states.row_mut(i).assign(&exp.next_state);
            dones[i] = exp.done;
        }
        let batch = ExperienceBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
            info: None,
        };
        Ok((batch, weights, chosen_indices))
    }

    /// Update priorities for the given indices with new TD-errors
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) -> Result<()> {
        for (&idx, &err) in indices.iter().zip(td_errors.iter()) {
            if idx < self.buffer.len() {
                let prio = err.abs() + 1e-6;
                self.buffer[idx].priority = prio;
                if prio > self.max_priority {
                    self.max_priority = prio;
                }
            }
        }
        Ok(())
    }

    /// Update the IS-weight exponent (typically annealed towards 1)
    pub fn update_beta(&mut self, beta: f32) {
        self.beta = beta.min(1.0);
    }

    /// Current number of stored experiences
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl ReplayBufferTrait for PrioritizedReplayBuffer {
    fn add(
        &mut self,
        state: Array1<f32>,
        action: Array1<f32>,
        reward: f32,
        next_state: Array1<f32>,
        done: bool,
    ) -> Result<()> {
        PrioritizedReplayBuffer::add(self, state, action, reward, next_state, done)
    }

    fn sample_batch(&self, _batch_size: usize) -> Result<ExperienceBatch> {
        Err(NeuralError::InvalidArgument(
            "Use PrioritizedReplayBuffer::sample(&mut self, batch_size) for mutable sampling"
                .to_string(),
        ))
    }

    fn len(&self) -> usize {
        PrioritizedReplayBuffer::len(self)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_experience(
        val: f32,
        state_dim: usize,
        action_dim: usize,
    ) -> (Array1<f32>, Array1<f32>, f32, Array1<f32>, bool) {
        (
            Array1::from_elem(state_dim, val),
            Array1::from_elem(action_dim, val * 0.1),
            val,
            Array1::from_elem(state_dim, val + 1.0),
            false,
        )
    }

    #[test]
    fn test_replay_buffer_add_sample() {
        let mut buf = ReplayBuffer::new(100);
        for i in 0..10 {
            let (s, a, r, ns, d) = make_experience(i as f32, 4, 2);
            buf.add(s, a, r, ns, d).expect("add ok");
        }
        assert_eq!(buf.len(), 10);
        let batch = buf.sample(5).expect("sample ok");
        assert_eq!(batch.states.shape(), &[5, 4]);
        assert_eq!(batch.actions.shape(), &[5, 2]);
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buf = ReplayBuffer::new(5);
        for i in 0..10 {
            let (s, a, r, ns, d) = make_experience(i as f32, 2, 1);
            buf.add(s, a, r, ns, d).expect("add ok");
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_replay_buffer_trait() {
        let buf: Box<dyn ReplayBufferTrait> = Box::new(ReplayBuffer::new(50));
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 50);
    }

    #[test]
    fn test_prioritized_replay_buffer() {
        let mut buf = PrioritizedReplayBuffer::new(100, 0.6, 0.4);
        for i in 0..20 {
            let (s, a, r, ns, d) = make_experience(i as f32, 4, 2);
            buf.add(s, a, r, ns, d).expect("add ok");
        }
        assert_eq!(buf.len(), 20);
        let (batch, weights, indices) = buf.sample(8).expect("sample ok");
        assert_eq!(batch.states.shape(), &[8, 4]);
        assert_eq!(weights.len(), 8);
        assert_eq!(indices.len(), 8);
        // Update priorities
        let td_errors: Vec<f32> = (0..8).map(|i| i as f32 * 0.1 + 0.01).collect();
        buf.update_priorities(&indices, &td_errors)
            .expect("update ok");
    }

    #[test]
    fn test_prioritized_buffer_beta_update() {
        let mut buf = PrioritizedReplayBuffer::new(10, 0.6, 0.4);
        buf.update_beta(0.9);
        assert!((buf.beta - 0.9).abs() < 1e-6);
        buf.update_beta(1.5); // clamped
        assert!((buf.beta - 1.0).abs() < 1e-6);
    }
}
