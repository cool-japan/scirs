//! Curiosity-driven exploration for reinforcement learning

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::prelude::*;
use scirs2_core::random::rng;
use std::collections::VecDeque;

// ── Internal sub-networks ─────────────────────────────────────────────────────

struct FeatureEncoder {
    layers: Vec<Box<dyn Layer<f32>>>,
    output_dim: usize,
}

impl FeatureEncoder {
    fn new(input_dim: usize, output_dim: usize, hidden_sizes: Vec<usize>) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current = input_dim;
        for &h in &hidden_sizes {
            layers.push(Box::new(Dense::new(current, h, Some("relu"), &mut rng())?));
            current = h;
        }
        layers.push(Box::new(Dense::new(current, output_dim, None, &mut rng())?));
        Ok(Self { layers, output_dim })
    }

    fn encode(&self, input: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut x: ArrayD<f32> = input.to_owned().insert_axis(Axis(0)).into_dyn();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x2 = x
            .into_dimensionality::<Ix2>()
            .map_err(|e| NeuralError::InvalidArgument(format!("encoder reshape: {e}")))?;
        Ok(x2.row(0).to_owned())
    }
}

struct ForwardModel {
    layers: Vec<Box<dyn Layer<f32>>>,
}

impl ForwardModel {
    fn new(
        feature_dim: usize,
        action_dim: usize,
        output_dim: usize,
        hidden_sizes: Vec<usize>,
    ) -> Result<Self> {
        let input_dim = feature_dim + action_dim;
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current = input_dim;
        for &h in &hidden_sizes {
            layers.push(Box::new(Dense::new(current, h, Some("relu"), &mut rng())?));
            current = h;
        }
        layers.push(Box::new(Dense::new(current, output_dim, None, &mut rng())?));
        Ok(Self { layers })
    }

    fn predict(&self, feature: &ArrayView1<f32>, action: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let mut input_vec: Vec<f32> = feature.iter().chain(action.iter()).cloned().collect();
        let input: ArrayD<f32> = Array2::from_shape_vec((1, input_vec.len()), input_vec)
            .map_err(|e| NeuralError::InvalidArgument(format!("forward model input: {e}")))?
            .into_dyn();
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x2 = x
            .into_dimensionality::<Ix2>()
            .map_err(|e| NeuralError::InvalidArgument(format!("forward model reshape: {e}")))?;
        Ok(x2.row(0).to_owned())
    }
}

struct InverseModel {
    layers: Vec<Box<dyn Layer<f32>>>,
}

impl InverseModel {
    fn new(
        feature_dim_1: usize,
        feature_dim_2: usize,
        output_dim: usize,
        hidden_sizes: Vec<usize>,
    ) -> Result<Self> {
        let input_dim = feature_dim_1 + feature_dim_2;
        let mut layers: Vec<Box<dyn Layer<f32>>> = Vec::new();
        let mut current = input_dim;
        for &h in &hidden_sizes {
            layers.push(Box::new(Dense::new(current, h, Some("relu"), &mut rng())?));
            current = h;
        }
        layers.push(Box::new(Dense::new(current, output_dim, None, &mut rng())?));
        Ok(Self { layers })
    }

    fn predict(&self, feat1: &ArrayView1<f32>, feat2: &ArrayView1<f32>) -> Result<Array1<f32>> {
        let input_vec: Vec<f32> = feat1.iter().chain(feat2.iter()).cloned().collect();
        let input: ArrayD<f32> = Array2::from_shape_vec((1, input_vec.len()), input_vec)
            .map_err(|e| NeuralError::InvalidArgument(format!("inverse model input: {e}")))?
            .into_dyn();
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x2 = x
            .into_dimensionality::<Ix2>()
            .map_err(|e| NeuralError::InvalidArgument(format!("inverse model reshape: {e}")))?;
        Ok(x2.row(0).to_owned())
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Intrinsic Curiosity Module (Pathak et al. 2017)
pub struct ICM {
    forward_model: ForwardModel,
    inverse_model: InverseModel,
    feature_encoder: FeatureEncoder,
    eta: f32,
    beta: f32,
}

impl ICM {
    /// Create a new ICM
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        feature_dim: usize,
        hidden_sizes: Vec<usize>,
        eta: f32,
        beta: f32,
    ) -> Result<Self> {
        let feature_encoder = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes.clone())?;
        let forward_model =
            ForwardModel::new(feature_dim, action_dim, feature_dim, hidden_sizes.clone())?;
        let inverse_model = InverseModel::new(feature_dim, feature_dim, action_dim, hidden_sizes)?;
        Ok(Self {
            forward_model,
            inverse_model,
            feature_encoder,
            eta,
            beta,
        })
    }

    /// Compute intrinsic reward = η × ||φ'(s') − φ̂'(s')||²
    pub fn compute_intrinsic_reward(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
        next_state: &ArrayView1<f32>,
    ) -> Result<f32> {
        let phi_s = self.feature_encoder.encode(state)?;
        let phi_s_next = self.feature_encoder.encode(next_state)?;
        let phi_s_next_pred = self.forward_model.predict(&phi_s.view(), action)?;
        let n = phi_s_next.len().min(phi_s_next_pred.len());
        let error = (0..n)
            .map(|i| (phi_s_next[i] - phi_s_next_pred[i]).powi(2))
            .sum::<f32>();
        Ok(self.eta / 2.0 * error)
    }

    /// Inverse model loss (predicts action from (φ(s), φ(s')))
    pub fn compute_inverse_loss(
        &self,
        state: &ArrayView1<f32>,
        next_state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
    ) -> Result<f32> {
        let phi_s = self.feature_encoder.encode(state)?;
        let phi_s_next = self.feature_encoder.encode(next_state)?;
        let pred_action = self
            .inverse_model
            .predict(&phi_s.view(), &phi_s_next.view())?;
        let n = action.len().min(pred_action.len());
        let loss = (0..n)
            .map(|i| (action[i] - pred_action[i]).powi(2))
            .sum::<f32>()
            / n.max(1) as f32;
        Ok(loss)
    }

    /// Combined ICM update loss
    pub fn compute_loss(
        &self,
        state: &ArrayView1<f32>,
        action: &ArrayView1<f32>,
        next_state: &ArrayView1<f32>,
    ) -> Result<f32> {
        let inv_loss = self.compute_inverse_loss(state, next_state, action)?;
        let phi_s = self.feature_encoder.encode(state)?;
        let phi_next = self.feature_encoder.encode(next_state)?;
        let phi_pred = self.forward_model.predict(&phi_s.view(), action)?;
        let n = phi_next.len().min(phi_pred.len());
        let fwd_loss = (0..n)
            .map(|i| (phi_next[i] - phi_pred[i]).powi(2))
            .sum::<f32>()
            / n.max(1) as f32;
        Ok((1.0 - self.beta) * inv_loss + self.beta * fwd_loss)
    }

    /// Scaling factor η
    pub fn eta(&self) -> f32 {
        self.eta
    }
}

/// Random Network Distillation exploration bonus (Burda et al. 2019)
pub struct RND {
    /// Fixed target network (random, never updated)
    target: FeatureEncoder,
    /// Predictor network (trained to match target)
    predictor: FeatureEncoder,
    /// Running mean/var for normalisation
    reward_mean: f32,
    reward_var: f32,
    reward_count: usize,
}

impl RND {
    /// Create a new RND module
    pub fn new(state_dim: usize, feature_dim: usize, hidden_sizes: Vec<usize>) -> Result<Self> {
        let target = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes.clone())?;
        let predictor = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes)?;
        Ok(Self {
            target,
            predictor,
            reward_mean: 0.0,
            reward_var: 1.0,
            reward_count: 0,
        })
    }

    /// Compute normalised intrinsic reward for `state`
    pub fn compute_intrinsic_reward(&mut self, state: &ArrayView1<f32>) -> Result<f32> {
        let target_feat = self.target.encode(state)?;
        let pred_feat = self.predictor.encode(state)?;
        let n = target_feat.len().min(pred_feat.len());
        let raw = (0..n)
            .map(|i| (target_feat[i] - pred_feat[i]).powi(2))
            .sum::<f32>();
        // Update running stats
        self.reward_count += 1;
        let delta = raw - self.reward_mean;
        self.reward_mean += delta / self.reward_count as f32;
        let delta2 = raw - self.reward_mean;
        self.reward_var += (delta * delta2 - self.reward_var) / self.reward_count as f32;
        // Normalise
        Ok(raw / (self.reward_var.sqrt().max(1e-8)))
    }
}

/// Episodic curiosity via a reachability buffer (Savinov et al. 2019)
pub struct EpisodicCuriosity {
    feature_encoder: FeatureEncoder,
    buffer: VecDeque<Array1<f32>>,
    buffer_capacity: usize,
    k_neighbours: usize,
    beta: f32,
}

impl EpisodicCuriosity {
    /// Create a new episodic curiosity module
    pub fn new(
        state_dim: usize,
        feature_dim: usize,
        hidden_sizes: Vec<usize>,
        buffer_capacity: usize,
        k_neighbours: usize,
        beta: f32,
    ) -> Result<Self> {
        let feature_encoder = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes)?;
        Ok(Self {
            feature_encoder,
            buffer: VecDeque::new(),
            buffer_capacity,
            k_neighbours,
            beta,
        })
    }

    /// Compute episodic intrinsic reward for `state`
    pub fn compute_reward(&mut self, state: &ArrayView1<f32>) -> Result<f32> {
        let phi = self.feature_encoder.encode(state)?;

        // Compute distances to k-nearest neighbours in the buffer
        let reward = if self.buffer.is_empty() {
            1.0 // Novel — first state in episode
        } else {
            let mut dists: Vec<f32> = self
                .buffer
                .iter()
                .map(|stored| {
                    let n = phi.len().min(stored.len());
                    (0..n)
                        .map(|i| (phi[i] - stored[i]).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).expect("non-NaN"));
            let k = self.k_neighbours.min(dists.len());
            let knn_mean = dists[..k].iter().sum::<f32>() / k.max(1) as f32;
            (knn_mean / (self.beta + knn_mean)).max(0.0)
        };

        // Add to buffer
        if self.buffer.len() >= self.buffer_capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(phi);
        Ok(reward)
    }

    /// Reset the episodic buffer (call at start of each episode)
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

/// Count-based novelty exploration heuristic
pub struct NoveltyExploration {
    feature_encoder: FeatureEncoder,
    visit_counts: std::collections::HashMap<Vec<i32>, usize>,
    resolution: f32,
}

impl NoveltyExploration {
    /// Create a new novelty exploration module
    pub fn new(
        state_dim: usize,
        feature_dim: usize,
        hidden_sizes: Vec<usize>,
        resolution: f32,
    ) -> Result<Self> {
        let feature_encoder = FeatureEncoder::new(state_dim, feature_dim, hidden_sizes)?;
        Ok(Self {
            feature_encoder,
            visit_counts: std::collections::HashMap::new(),
            resolution,
        })
    }

    /// Get intrinsic reward based on visit count
    pub fn intrinsic_reward(&mut self, state: &ArrayView1<f32>) -> Result<f32> {
        let phi = self.feature_encoder.encode(state)?;
        let key: Vec<i32> = phi
            .iter()
            .map(|&x| (x / self.resolution).round() as i32)
            .collect();
        let count = self.visit_counts.entry(key).or_insert(0);
        *count += 1;
        Ok(1.0 / (*count as f32).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icm_intrinsic_reward() {
        let icm = ICM::new(4, 2, 8, vec![16], 0.01, 0.2).expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::from_vec(vec![1.0, 0.0]);
        let next_state = Array1::ones(4);
        let r = icm
            .compute_intrinsic_reward(&state.view(), &action.view(), &next_state.view())
            .expect("reward ok");
        assert!(r.is_finite());
        assert!(r >= 0.0);
    }

    #[test]
    fn test_icm_inverse_loss() {
        let icm = ICM::new(4, 2, 8, vec![16], 0.01, 0.2).expect("create ok");
        let state = Array1::zeros(4);
        let next_state = Array1::ones(4);
        let action = Array1::from_vec(vec![0.5, 0.5]);
        let loss = icm
            .compute_inverse_loss(&state.view(), &next_state.view(), &action.view())
            .expect("loss ok");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_rnd_reward() {
        let mut rnd = RND::new(4, 8, vec![16]).expect("create ok");
        let state = Array1::zeros(4);
        let r = rnd
            .compute_intrinsic_reward(&state.view())
            .expect("reward ok");
        assert!(r.is_finite());
    }

    #[test]
    fn test_episodic_curiosity_reward() {
        let mut ec = EpisodicCuriosity::new(4, 8, vec![16], 100, 5, 0.001).expect("create ok");
        let state1 = Array1::zeros(4);
        let r1 = ec.compute_reward(&state1.view()).expect("reward ok");
        assert!(r1.is_finite());
        let state2 = Array1::ones(4);
        let r2 = ec.compute_reward(&state2.view()).expect("reward ok");
        assert!(r2.is_finite());
    }

    #[test]
    fn test_novelty_exploration() {
        let mut ne = NoveltyExploration::new(4, 8, vec![16], 0.1).expect("create ok");
        let state = Array1::zeros(4);
        let r1 = ne.intrinsic_reward(&state.view()).expect("reward ok");
        let r2 = ne.intrinsic_reward(&state.view()).expect("reward ok");
        assert!(
            r1 > r2,
            "reward should decrease with more visits: {r1} > {r2}"
        );
    }

    #[test]
    fn test_icm_combined_loss() {
        let icm = ICM::new(4, 2, 8, vec![16], 0.01, 0.2).expect("create ok");
        let state = Array1::zeros(4);
        let action = Array1::from_vec(vec![1.0, 0.0]);
        let next_state = Array1::ones(4);
        let loss = icm
            .compute_loss(&state.view(), &action.view(), &next_state.view())
            .expect("loss ok");
        assert!(loss.is_finite());
    }
}
