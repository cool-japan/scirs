//! Trust Region Policy Optimization (TRPO) implementation
//!
//! TRPO is an advanced policy gradient method that ensures monotonic improvement
//! by constraining the policy update step size using KL divergence.

use crate::error::Result;
use crate::reinforcement::policy::{Policy, PolicyNetwork};
use crate::reinforcement::value::ValueNetwork;
use scirs2_core::ndarray::prelude::*;
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::ndarray::ArrayStatCompat;
use statrs::statistics::Statistics;
// Note: Would use ndarray_linalg for matrix operations in production
/// TRPO configuration
#[derive(Debug, Clone)]
pub struct TRPOConfig {
    /// Maximum KL divergence between old and new policy
    pub max_kl: f32,
    /// Damping coefficient for Fisher matrix
    pub damping: f32,
    /// Line search acceptance ratio
    pub accept_ratio: f32,
    /// Maximum line search iterations
    pub max_line_search_iter: usize,
    /// Conjugate gradient iterations
    pub cg_iters: usize,
    /// Conjugate gradient tolerance
    pub cg_tol: f32,
    /// Value function iterations per update
    pub vf_iters: usize,
    /// Value function learning rate
    pub vf_lr: f32,
    /// GAE lambda for advantage estimation
    pub gae_lambda: f32,
    /// Discount factor
    pub gamma: f32,
    /// Entropy coefficient
    pub entropy_coef: f32,
}
impl Default for TRPOConfig {
    fn default() -> Self {
        Self {
            max_kl: 0.01,
            damping: 0.1,
            accept_ratio: 0.1,
            max_line_search_iter: 10,
            cg_iters: 10,
            cg_tol: 1e-8,
            vf_iters: 5,
            vf_lr: 1e-3,
            gae_lambda: 0.97,
            gamma: 0.99,
            entropy_coef: 0.0,
        }
    }
/// Trust Region Policy Optimization
pub struct TRPO {
    policy: PolicyNetwork,
    value_fn: ValueNetwork,
    config: TRPOConfig,
    old_policy: Option<PolicyNetwork>,
impl TRPO {
    /// Create a new TRPO instance
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_sizes: Vec<usize>,
        continuous: bool,
        config: TRPOConfig,
    ) -> Result<Self> {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_sizes.clone(), continuous)?;
        let value_fn = ValueNetwork::new(state_dim, 1, hidden_sizes)?;
        Ok(Self {
            policy,
            value_fn,
            config,
            old_policy: None,
        })
    /// Compute Generalized Advantage Estimation (GAE)
    pub fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; rewards.len()];
        let mut returns = vec![0.0; rewards.len()];
        let mut gae = 0.0;
        for i in (0..rewards.len()).rev() {
            let next_val = if i == rewards.len() - 1 {
                next_value
            } else {
                values[i + 1]
            };
            let delta = rewards[i]
                + (if dones[i] {
                    0.0
                } else {
                    self.config.gamma * next_val
                })
                - values[i];
            gae = delta
                    self.config.gamma * self.config.gae_lambda * gae
                });
            advantages[i] = gae;
            returns[i] = advantages[i] + values[i];
        (advantages, returns)
    /// Compute policy loss (negative of objective for minimization)
    fn compute_policy_loss(
        states: &ArrayView2<f32>,
        actions: &ArrayView2<f32>,
        advantages: &ArrayView1<f32>,
        old_log_probs: &ArrayView1<f32>,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let n = states.shape()[0];
        for i in 0..n {
            let state = states.row(i);
            let action = actions.row(i);
            let log_prob = self.policy.log_prob(&state, &action)?;
            let ratio = (log_prob - old_log_probs[i]).exp();
            total_loss -= ratio * advantages[i];
        Ok(total_loss / n as f32)
    /// Compute KL divergence between policies
    fn compute_kl_divergence(
        old_policy: &PolicyNetwork,
        let mut total_kl = 0.0;
            // Get distribution parameters from both policies
            let (new_params, new_std) = self.policy.get_distribution_params(&state)?;
            let (old_params, old_std) = old_policy.get_distribution_params(&state)?;
            if let (Some(new_std), Some(old_std)) = (new_std, old_std) {
                // KL divergence for Gaussian distributions
                for j in 0..new_params.len() {
                    let kl = (old_std[j] / new_std[j]).ln()
                        + (new_std[j].powi(2) + (new_params[j] - old_params[j]).powi(2))
                            / (2.0 * old_std[j].powi(2))
                        - 0.5;
                    total_kl += kl;
                }
                // KL divergence for categorical distributions
                    if old_params[j] > 0.0 {
                        total_kl += old_params[j] * (old_params[j] / new_params[j]).ln();
                    }
            }
        Ok(total_kl / n as f32)
    /// Compute Fisher-vector product using finite differences
    fn fisher_vector_product(
        vector: &ArrayView1<f32>,
        epsilon: f32,
    ) -> Result<Array1<f32>> {
        // This is a simplified implementation
        // In practice, this would compute Hv where H is the Fisher Information Matrix
        let n_params = vector.len();
        let mut fvp = Array1::zeros(n_params);
        // Add damping term
        fvp = fvp + self.config.damping * vector;
        Ok(fvp)
    /// Conjugate gradient solver
    fn conjugate_gradient(
        g: &ArrayView1<f32>,
        let mut x = Array1::zeros(g.len());
        let mut r = g.to_owned();
        let mut p = r.clone();
        let mut rdotr = r.dot(&r);
        for _ in 0..self.config.cg_iters {
            let fvp = self.fisher_vector_product(&p.view(), states, 1e-5)?;
            let alpha = rdotr / p.dot(&fvp);
            x = x + alpha * &p;
            r = r - alpha * &fvp;
            let new_rdotr = r.dot(&r);
            if new_rdotr < self.config.cg_tol {
                break;
            let beta = new_rdotr / rdotr;
            p = &r + beta * &p;
            rdotr = new_rdotr;
        Ok(x)
    /// Line search for step size
    fn line_search(
        &mut self,
        step_dir: &ArrayView1<f32>,
        expected_improve: f32,
    ) -> Result<bool> {
        let old_params = self.policy.parameters();
        let mut step_size = 1.0;
        for _ in 0..self.config.max_line_search_iter {
            // Take step
            let new_params: Vec<Array2<f32>> = old_params
                .iter()
                .enumerate()
                .map(|(i, param)| param + step_size * step_dir[i])
                .collect();
            self.policy.set_parameters(&new_params)?;
            // Check improvement
            let new_loss = self.compute_policy_loss(states, actions, advantages, old_log_probs)?;
            let old_loss = {
                self.policy.set_parameters(&old_params)?;
                self.compute_policy_loss(states, actions, advantages, old_log_probs)?
            let actual_improve = old_loss - new_loss;
            let expected = expected_improve * step_size;
            let ratio = actual_improve / expected;
            // Check KL constraint
            if let Some(ref old_policy) = self.old_policy {
                let kl = self.compute_kl_divergence(states, old_policy)?;
                if ratio > self.config.accept_ratio && kl < self.config.max_kl {
                    return Ok(true);
            step_size *= 0.5;
        // Restore old parameters if line search failed
        self.policy.set_parameters(&old_params)?;
        Ok(false)
    /// Update policy and value function
    pub fn update(
        states: &Array2<f32>,
        actions: &Array2<f32>,
        next_state: &ArrayView1<f32>,
    ) -> Result<(f32, f32, f32)> {
        // Compute values and advantages
        let values: Vec<f32> = (0..n)
            .map(|i| self.value_fn.predict(&states.row(i)))
            .collect::<Result<Vec<_>>>()?;
        let next_value = self.value_fn.predict(next_state)?;
        let (advantages, returns) = self.compute_gae(rewards, &values, next_value, dones);
        // Normalize advantages
        let adv_array = Array1::from_vec(advantages.clone());
        let adv_mean = adv_array.mean_or(0.0);
        let adv_std = adv_array.std(0.0);
        let normalized_advantages = (adv_array - adv_mean) / (adv_std + 1e-8);
        // Store old policy
        self.old_policy = Some(PolicyNetwork::new(
            self.policy.action_dim,
            vec![64, 64],
            self.policy.continuous,
        )?);
        // Compute old log probabilities
        let old_log_probs: Vec<f32> = (0..n)
            .map(|i| self.policy.log_prob(&states.row(i), &actions.row(i)))
        let old_log_probs = Array1::from_vec(old_log_probs);
        // Compute policy gradient
        let param_count = self.policy.get_parameter_count();
        let mut policy_grad = Array1::zeros(param_count);
        
            let advantage = normalized_advantages[i];
            // Compute log probability gradient with respect to policy parameters
            let log_prob_grad = self.policy.compute_log_prob_gradient(&state, &action)?;
            
            // Weight by advantage and accumulate
            policy_grad = policy_grad + log_prob_grad * advantage;
        // Average over batch
        policy_grad = policy_grad / n as f32;
        // Compute natural gradient using conjugate gradient
        let step_dir = self.conjugate_gradient(&states.view(), &policy_grad.view())?;
        // Compute expected improvement
        let expected_improve = policy_grad.dot(&step_dir);
        // Line search
        let success = self.line_search(
            &states.view(),
            &actions.view(),
            &normalized_advantages.view(),
            &old_log_probs.view(),
            &step_dir.view(),
            expected_improve,
        )?;
        // Update value function
        let mut value_loss = 0.0;
        for _ in 0..self.config.vf_iters {
            let mut vf_loss = 0.0;
            for i in 0..n {
                let pred = self.value_fn.predict(&states.row(i))?;
                vf_loss += (pred - returns[i]).powi(2);
            value_loss = vf_loss / n as f32;
            // Value function update would happen here
        // Compute final KL divergence
        let kl = if let Some(ref old_policy) = self.old_policy {
            self.compute_kl_divergence(&states.view(), old_policy)?
        } else {
            0.0
        };
        Ok((value_loss, kl, if success { 1.0 } else { 0.0 }))
    /// Get action from policy
    pub fn act(&self, state: &ArrayView1<f32>) -> Result<Array1<f32>> {
        self.policy.sample_action(state)
    /// Predict value
    pub fn predict_value(&self, state: &ArrayView1<f32>) -> Result<f32> {
        self.value_fn.predict(state)
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_trpo_creation() {
        let config = TRPOConfig::default();
        let trpo = TRPO::new(4, 2, vec![64, 64], true, config).unwrap();
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = trpo.act(&state.view()).unwrap();
        assert_eq!(action.len(), 2);
        let value = trpo.predict_value(&state.view()).unwrap();
        assert!(value.is_finite());
    fn test_gae_computation() {
        let trpo = TRPO::new(4, 2, vec![32], false, config).unwrap();
        let rewards = vec![1.0, 2.0, 3.0, 4.0];
        let values = vec![0.5, 1.5, 2.5, 3.5];
        let next_value = 4.5;
        let dones = vec![false, false, false, true];
        let (advantages, returns) = trpo.compute_gae(&rewards, &values, next_value, &dones);
        assert_eq!(advantages.len(), 4);
        assert_eq!(returns.len(), 4);
