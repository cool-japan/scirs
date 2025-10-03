//! Advanced federated learning algorithms
//!
//! This module implements state-of-the-art federated learning algorithms
//! including SCAFFOLD, FedAvgM, and others.

use crate::error::{NeuralError, Result};
use crate::federated::{AggregationStrategy, ClientUpdate};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
/// SCAFFOLD algorithm - stochastic controlled averaging for federated learning
pub struct SCAFFOLD {
    /// Server control variate
    server_control: Option<Vec<Array2<f32>>>,
    /// Client control variates
    client_controls: HashMap<usize, Vec<Array2<f32>>>,
    /// Learning rate for server
    server_lr: f32,
    /// Global learning rate
    global_lr: f32,
}
impl SCAFFOLD {
    /// Create new SCAFFOLD aggregator
    pub fn new(_server_lr: f32, globallr: f32) -> Self {
        Self {
            server_control: None,
            client_controls: HashMap::new(),
            server_lr,
            global_lr,
        }
    }
    /// Update client control variate
    pub fn update_client_control(
        &mut self,
        client_id: usize,
        old_weights: &[Array2<f32>],
        new_weights: &[Array2<f32>],
        local_steps: usize,
        local_lr: f32,
    ) -> Result<()> {
        let mut new_control = Vec::new();
        if let Some(old_control) = self.client_controls.get(&client_id) {
            // c_i^+ = c_i - c + 1/(K*lr) * (x_i - y_i)
            let server_control = self.server_control.as_ref().unwrap_or(&vec![]);
            for (idx, ((old_w, new_w), old_c)) in old_weights
                .iter()
                .zip(newweights.iter())
                .zip(old_control.iter())
                .enumerate()
            {
                let server_c = server_control
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros(old_w.shape()));
                let gradient_term = (old_w - new_w) / (local_steps as f32 * local_lr);
                let new_c = old_c - &server_c + &gradient_term;
                new_control.push(new_c);
            }
        } else {
            // Initialize control variate for new client
            for (old_w, new_w) in oldweights.iter().zip(newweights.iter()) {
                new_control.push(gradient_term);
        self.client_controls.insert(client_id, new_control);
        Ok(())
    /// Update server control variate
    fn update_server_control(&mut self, clientupdates: &[ClientUpdate]) -> Result<()> {
        if client_updates.is_empty() {
            return Ok(());
        let num_tensors = client_updates[0].weight_updates.len();
        let mut new_server_control = Vec::new();
        for tensor_idx in 0..num_tensors {
            let shape = client_updates[0].weight_updates[tensor_idx].shape();
            let mut control_sum = Array2::zeros((shape[0], shape[1]));
            let mut total_samples = 0;
            // Average client control variates
            for update in client_updates {
                if let Some(client_control) = self.client_controls.get(&update.client_id) {
                    if tensor_idx < client_control.len() {
                        control_sum =
                            control_sum + &client_control[tensor_idx] * update.num_samples as f32;
                        total_samples += update.num_samples;
                    }
                }
            if total_samples > 0 {
                control_sum /= total_samples as f32;
            new_server_control.push(control_sum);
        self.server_control = Some(new_server_control);
impl AggregationStrategy for SCAFFOLD {
    fn aggregate(&mut self, updates: &[ClientUpdate], weights: &[f32]) -> Result<Vec<Array2<f32>>> {
        if updates.is_empty() {
            return Ok(Vec::new());
        // Update server control variate
        self.update_server_control(updates)?;
        let num_tensors = updates[0].weight_updates.len();
        let mut aggregated = Vec::with_capacity(num_tensors);
            let shape = updates[0].weight_updates[tensor_idx].shape();
            let mut weighted_sum = Array2::zeros((shape[0], shape[1]));
            // Weighted aggregation of client updates
            for (update, &weight) in updates.iter().zip(weights.iter()) {
                if tensor_idx < update.weight_updates.len() {
                    weighted_sum = weighted_sum + weight * &update.weight_updates[tensor_idx];
            // Apply control variate correction
            if let Some(ref server_control) = self.server_control {
                if tensor_idx < server_control.len() {
                    weighted_sum = weighted_sum + &server_control[tensor_idx] * self.global_lr;
            aggregated.push(weighted_sum * self.server_lr);
        Ok(aggregated)
    fn name(&self) -> &str {
        "SCAFFOLD"
/// FedAvgM - FedAvg with server momentum
pub struct FedAvgM {
    /// Server momentum parameter
    momentum: f32,
    /// Server learning rate
    /// Momentum buffers
    momentum_buffers: Option<Vec<Array2<f32>>>,
impl FedAvgM {
    /// Create new FedAvgM aggregator
    pub fn new(_serverlr: f32, momentum: f32) -> Self {
            momentum,
            momentum_buffers: None,
impl AggregationStrategy for FedAvgM {
        // First compute weighted average
            aggregated.push(weighted_sum);
        // Apply server momentum
        if self.momentum_buffers.is_none() {
            // Initialize momentum buffers
            self.momentum_buffers = Some(
                aggregated
                    .iter()
                    .map(|a| Array2::zeros(a.shape()))
                    .collect(),
            );
        if let Some(ref mut buffers) = self.momentum_buffers {
            for (i, (update, buffer)) in aggregated.iter_mut().zip(buffers.iter_mut()).enumerate() {
                // v_t = momentum * v_{t-1} + lr * update_t
                *buffer = &*buffer * self.momentum + &*update * self.server_lr;
                *update = buffer.clone();
        "FedAvgM"
/// FedAdam - Adaptive federated optimization
pub struct FedAdam {
    /// Learning rate
    lr: f32,
    /// First moment decay rate
    beta1: f32,
    /// Second moment decay rate
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// First moment estimates
    m: Option<Vec<Array2<f32>>>,
    /// Second moment estimates
    v: Option<Vec<Array2<f32>>>,
    /// Step counter
    step: usize,
impl FedAdam {
    /// Create new FedAdam aggregator
    pub fn new(lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
            lr,
            beta1,
            beta2,
            epsilon,
            m: None,
            v: None,
            step: 0,
impl AggregationStrategy for FedAdam {
        // Increment step counter
        self.step += 1;
        // First compute weighted average (gradient)
        let mut gradients = Vec::new();
            gradients.push(weighted_sum);
        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(gradients.iter().map(|g| Array2::zeros(g.shape())).collect());
            self.v = Some(gradients.iter().map(|g| Array2::zeros(g.shape())).collect());
        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        // Update moment estimates and compute updates
        for (i, grad) in gradients.into_iter().enumerate() {
            // Update biased first moment estimate
            m[i] = &m[i] * self.beta1 + &grad * (1.0 - self.beta1);
            // Update biased second moment estimate
            v[i] = &v[i] * self.beta2 + &grad * &grad * (1.0 - self.beta2);
            // Compute bias-corrected moment estimates
            let m_hat = &m[i] / (1.0 - self.beta1.powi(self.step as i32));
            let v_hat = &v[i] / (1.0 - self.beta2.powi(self.step as i32));
            // Compute update
            let update = &m_hat * self._lr / (v_hat.mapv(f32::sqrt) + self.epsilon);
            aggregated.push(update);
        "FedAdam"
/// FedAdagrad - Adaptive gradient algorithm for federated learning
pub struct FedAdagrad {
    /// Accumulated squared gradients
    acc_grad: Option<Vec<Array2<f32>>>,
impl FedAdagrad {
    /// Create new FedAdagrad aggregator
    pub fn new(lr: f32, epsilon: f32) -> Self {
            acc_grad: None,
impl AggregationStrategy for FedAdagrad {
        // Initialize accumulated gradients if needed
        if self.acc_grad.is_none() {
            self.acc_grad = Some(gradients.iter().map(|g| Array2::zeros(g.shape())).collect());
        let acc_grad = self.acc_grad.as_mut().unwrap();
        // Update accumulated gradients and compute updates
            // Accumulate squared gradients
            acc_grad[i] = &acc_grad[i] + &grad * &grad;
            // Compute adaptive learning rate
            let adaptive_lr = &acc_grad[i].mapv(f32::sqrt) + self.epsilon;
            let update = &grad * self._lr / adaptive_lr;
        "FedAdagrad"
/// FedLAG - Federated learning with Lookahead and Gradient tracking
pub struct FedLAG {
    /// Slow weights update frequency
    k: usize,
    /// Slow weights step size
    alpha: f32,
    /// Fast weights (current model)
    fast_weights: Option<Vec<Array2<f32>>>,
    /// Slow weights (lookahead weights)
    slow_weights: Option<Vec<Array2<f32>>>,
    step_count: usize,
impl FedLAG {
    /// Create new FedLAG aggregator
    pub fn new(k: usize, alpha: f32) -> Self {
            k,
            alpha,
            fast_weights: None,
            slow_weights: None,
            step_count: 0,
impl AggregationStrategy for FedLAG {
        // Standard weighted aggregation
        // Initialize weights if needed
        if self.fastweights.is_none() {
            self.fast_weights = Some(aggregated.clone());
            self.slow_weights = Some(aggregated.clone());
        // Update fast weights
        if let Some(ref mut fast_weights) = self.fast_weights {
            for (fast_w, update) in fastweights.iter_mut().zip(&aggregated) {
                *fast_w = &*fast_w + update;
        self.step_count += 1;
        // Update slow weights every k steps
        if self.step_count % self.k == 0 {
            if let (Some(ref mut slow_weights), Some(ref fast_weights)) =
                (&mut self.slow_weights, &self.fast_weights)
                for (slow_w, fast_w) in slowweights.iter_mut().zip(fast_weights) {
                    *slow_w = &*slow_w + &(*fast_w - &*slow_w) * self.alpha;
                // Reset fast weights to slow weights
                if let Some(ref mut fast_weights) = self.fast_weights {
                    for (fast_w, slow_w) in fastweights.iter_mut().zip(slow_weights) {
                        *fast_w = slow_w.clone();
        // Return the current fast weights update
        "FedLAG"
/// FedProx - Federated optimization with proximal term
pub struct FedProx {
    /// Proximal term coefficient
    mu: f32,
    /// Global model weights (for proximal term)
    global_weights: Option<Vec<Array2<f32>>>,
impl FedProx {
    /// Create new FedProx aggregator
    pub fn new(mu: f32) -> Self {
            mu,
            global_weights: None,
    /// Update global weights
    pub fn update_global_weights(&mut self, weights: Vec<Array2<f32>>) {
        self.global_weights = Some(weights);
impl AggregationStrategy for FedProx {
            // Apply proximal term if global weights are available
            if let Some(ref global_weights) = self.global_weights {
                if tensor_idx < globalweights.len() {
                    // Proximal term: mu * (w - w_global)
                    let proximal_term = &weighted_sum * self.mu;
                    weighted_sum = weighted_sum - proximal_term;
        "FedProx"
/// FedNova - Normalized averaging for heterogeneous federated learning
pub struct FedNova {
    /// Momentum parameter
    /// Server momentum buffer
    server_momentum: Option<Vec<Array2<f32>>>,
    /// Effective local steps per client
    effective_steps: HashMap<usize, f32>,
impl FedNova {
    /// Create new FedNova aggregator
    pub fn new(momentum: f32) -> Self {
            server_momentum: None,
            effective_steps: HashMap::new(),
    /// Update effective steps for a client
    pub fn update_effective_steps(&mut self, clientid: usize, steps: f32) {
        self.effective_steps.insert(client_id, steps);
impl AggregationStrategy for FedNova {
        // Calculate normalization factor based on effective local steps
        let mut total_norm_factor = 0.0;
        let mut client_norm_factors = Vec::new();
        for update in updates {
            let effective_steps = self.effective_steps.get(&update.client_id).copied().unwrap_or(1.0);
            let norm_factor = (update.num_samples as f32) / effective_steps;
            client_norm_factors.push(norm_factor);
            total_norm_factor += norm_factor;
        // Normalize weights based on effective steps
            for (i, update) in updates.iter().enumerate() {
                    let normalized_weight = client_norm_factors[i] / total_norm_factor;
                    weighted_sum = weighted_sum + normalized_weight * &update.weight_updates[tensor_idx];
        if self.server_momentum.is_none() {
            self.server_momentum = Some(
                aggregated.iter().map(|a| Array2::zeros(a.shape())).collect(),
        if let Some(ref mut momentum_buffer) = self.server_momentum {
            for (update, buffer) in aggregated.iter_mut().zip(momentum_buffer.iter_mut()) {
                *buffer = &*buffer * self.momentum + &*update;
        "FedNova"
/// FedOpt - Adaptive server optimization for federated learning
pub struct FedOpt {
    /// Server optimizer type
    optimizer_type: String,
    /// Beta1 for Adam-like optimizers
    /// Beta2 for Adam-like optimizers
impl FedOpt {
    /// Create new FedOpt aggregator
    pub fn new(_optimizertype: String, lr: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
            optimizer_type,
impl AggregationStrategy for FedOpt {
        // First compute weighted average (pseudo-gradient)
        let mut pseudo_gradients = Vec::new();
            pseudo_gradients.push(weighted_sum);
        match self.optimizer_type.as_str() {
            "sgd" => {
                // Simple SGD
                for grad in pseudo_gradients {
                    aggregated.push(grad * self._lr);
            "adam" => {
                // Adam optimizer
                if self.m.is_none() {
                    self.m = Some(pseudo_gradients.iter().map(|g| Array2::zeros(g.shape())).collect());
                    self.v = Some(pseudo_gradients.iter().map(|g| Array2::zeros(g.shape())).collect());
                let m = self.m.as_mut().unwrap();
                let v = self.v.as_mut().unwrap();
                for (i, grad) in pseudo_gradients.into_iter().enumerate() {
                    // Update biased first moment estimate
                    m[i] = &m[i] * self.beta1 + &grad * (1.0 - self.beta1);
                    // Update biased second moment estimate
                    v[i] = &v[i] * self.beta2 + &grad * &grad * (1.0 - self.beta2);
                    // Compute bias-corrected moment estimates
                    let m_hat = &m[i] / (1.0 - self.beta1.powi(self.step as i32));
                    let v_hat = &v[i] / (1.0 - self.beta2.powi(self.step as i32));
                    // Compute update
                    let update = &m_hat * self._lr / (v_hat.mapv(f32::sqrt) + self.epsilon);
                    aggregated.push(update);
            "yogi" => {
                // Yogi optimizer (variant of Adam)
                    // Update first moment
                    // Update second moment (Yogi-style)
                    let grad_squared = &grad * &grad;
                    let v_diff = &grad_squared - &v[i];
                    let sign_matrix = v_diff.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 });
                    v[i] = &v[i] + &v_diff * &sign_matrix * (1.0 - self.beta2);
                    // Compute bias-corrected estimates
            _ => {
                return Err(NeuralError::InvalidArgument(format!(
                    "Unknown optimizer type: {}",
                    self.optimizer_type
                )));
        "FedOpt"
/// MIME - Mutual Information-based Model Enhancement for federated learning
pub struct MIME {
    /// Previous round's model weights
    prev_weights: Option<Vec<Array2<f32>>>,
    /// Control variate for variance reduction
    control_variate: Option<Vec<Array2<f32>>>,
    /// Regularization parameter
    lambda: f32,
impl MIME {
    /// Create new MIME aggregator
    pub fn new(_serverlr: f32, lambda: f32) -> Self {
            prev_weights: None,
            control_variate: None,
            lambda,
impl AggregationStrategy for MIME {
        let mut current_pseudo_grad = Vec::new();
            current_pseudo_grad.push(weighted_sum);
        // Initialize control variate if needed
        if self.control_variate.is_none() {
            self.control_variate = Some(
                current_pseudo_grad.iter().map(|g| Array2::zeros(g.shape())).collect(),
        // Apply MIME correction
        if let Some(ref mut control_variate) = self.control_variate {
            for (i, current_grad) in current_pseudo_grad.into_iter().enumerate() {
                // MIME update: current_grad + lambda * (current_grad - control_variate)
                let correction = &current_grad - &control_variate[i];
                let mime_update = &current_grad + &correction * self.lambda;
                // Update control variate
                control_variate[i] = current_grad.clone();
                // Apply server learning rate
                aggregated.push(mime_update * self._server_lr);
        // Update previous weights
        self.prev_weights = Some(aggregated.clone());
        "MIME"
/// Information about an aggregator algorithm
#[derive(Debug, Clone)]
pub struct AggregatorInfo {
    pub name: String,
    pub description: String,
    pub key_features: Vec<String>,
    pub recommended_use: String,
/// Aggregator factory for creating different algorithms
pub struct AggregatorFactory;
impl AggregatorFactory {
    /// Create aggregator by name
    pub fn create(
        name: &str,
        config: &HashMap<String, f32>,
    ) -> Result<Box<dyn AggregationStrategy>> {
        match name.to_lowercase().as_str() {
            "scaffold" => {
                let _server_lr = config.get("_server_lr").copied().unwrap_or(1.0);
                let global_lr = config.get("global_lr").copied().unwrap_or(1.0);
                Ok(Box::new(SCAFFOLD::new(_server_lr, global_lr)))
            "fedavgm" => {
                let momentum = config.get("momentum").copied().unwrap_or(0.9);
                Ok(Box::new(FedAvgM::new(_server_lr, momentum)))
            "fedadam" => {
                let lr = config.get("lr").copied().unwrap_or(0.001);
                let beta1 = config.get("beta1").copied().unwrap_or(0.9);
                let beta2 = config.get("beta2").copied().unwrap_or(0.999);
                let epsilon = config.get("epsilon").copied().unwrap_or(1e-8);
                Ok(Box::new(FedAdam::new(lr, beta1, beta2, epsilon)))
            "fedadagrad" => {
                let lr = config.get("lr").copied().unwrap_or(0.01);
                Ok(Box::new(FedAdagrad::new(lr, epsilon)))
            "fedlag" => {
                let k = config.get("k").copied().unwrap_or(5.0) as usize;
                let alpha = config.get("alpha").copied().unwrap_or(0.5);
                Ok(Box::new(FedLAG::new(k, alpha)))
            "fedprox" => {
                let mu = config.get("mu").copied().unwrap_or(0.01);
                Ok(Box::new(FedProx::new(mu)))
            "fednova" => {
                Ok(Box::new(FedNova::new(momentum)))
            "fedopt" => {
                let optimizer_type = config.get("optimizer_type").map(|&v| format!("{}", v as u32)).unwrap_or_else(|| "adam".to_string());
                
                // Handle optimizer type mapping
                let opt_type = match optimizer_type.as_str() {
                    "0" | "sgd" => "sgd".to_string(),
                    "1" | "adam" => "adam".to_string(),
                    "2" | "yogi" => "yogi".to_string(, _ => "adam".to_string(),
                };
                Ok(Box::new(FedOpt::new(opt_type, lr, beta1, beta2, epsilon)))
            "mime" => {
                let lambda = config.get("lambda").copied().unwrap_or(0.1);
                Ok(Box::new(MIME::new(server_lr, lambda)))
            _ => Err(NeuralError::InvalidArgument(format!(
                "Unknown aggregator: {}",
                name
            ))),
    /// Get list of available aggregators
    pub fn available_aggregators() -> Vec<&'static str> {
        vec!["scaffold", "fedavgm", "fedadam", "fedadagrad", "fedlag", "fedprox", "fednova", "fedopt", "mime"]
    /// Get detailed aggregator information
    #[allow(dead_code)]
    pub fn get_aggregator_info(name: &str) -> Option<AggregatorInfo> {
            "scaffold" => Some(AggregatorInfo {
                name: "SCAFFOLD".to_string(),
                description: "Stochastic Controlled Averaging for federated learning".to_string(),
                key_features: vec!["Control variates".to_string(), "Variance reduction".to_string()],
                recommended_use: "Heterogeneous data distributions".to_string(),
            }),
            "fedavgm" => Some(AggregatorInfo {
                name: "FedAvgM".to_string(),
                description: "FedAvg with server momentum".to_string(),
                key_features: vec!["Server momentum".to_string(), "Improved convergence".to_string()],
                recommended_use: "General federated learning".to_string(),
            "fedadam" => Some(AggregatorInfo {
                name: "FedAdam".to_string(),
                description: "Adaptive federated optimization with Adam".to_string(),
                key_features: vec!["Adaptive learning rates".to_string(), "Second-order moments".to_string()],
                recommended_use: "Tasks requiring adaptive optimization".to_string(, _ => None,
    /// Get default configuration for an aggregator
    pub fn default_config(name: &str) -> HashMap<String, f32> {
        let mut config = HashMap::new();
                config.insert("server_lr".to_string(), 1.0);
                config.insert("global_lr".to_string(), 1.0);
                config.insert("momentum".to_string(), 0.9);
                config.insert("lr".to_string(), 0.001);
                config.insert("beta1".to_string(), 0.9);
                config.insert("beta2".to_string(), 0.999);
                config.insert("epsilon".to_string(), 1e-8);
                config.insert("lr".to_string(), 0.01);
                config.insert("k".to_string(), 5.0);
                config.insert("alpha".to_string(), 0.5);
                config.insert("mu".to_string(), 0.01);
                config.insert("optimizer_type".to_string(), 1.0); // 1 = adam
                config.insert("lambda".to_string(), 0.1);
            _ => {}
        config
#[cfg(test)]
mod tests {
    use super::*;
    use crate::federated::ClientUpdate;
    fn create_test_updates() -> Vec<ClientUpdate> {
        vec![
            ClientUpdate {
                client_id: 0,
                weight_updates: vec![Array2::ones((3, 3))],
                num_samples: 100,
                loss: 0.5,
                accuracy: 0.9,
            },
                client_id: 1,
                weight_updates: vec![Array2::ones((3, 3)) * 2.0],
                num_samples: 200,
                loss: 0.4,
                accuracy: 0.92,
        ]
    #[test]
    fn test_scaffold() {
        let mut scaffold = SCAFFOLD::new(1.0, 1.0);
        let updates = create_test_updates();
        let weights = vec![0.5, 0.5];
        let result = scaffold.aggregate(&updates, &weights).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape(), &[3, 3]);
    fn test_fedavgm() {
        let mut fedavgm = FedAvgM::new(1.0, 0.9);
        let result = fedavgm.aggregate(&updates, &weights).unwrap();
    fn test_aggregator_factory() {
        let config = AggregatorFactory::default_config("fedadam");
        let aggregator = AggregatorFactory::create("fedadam", &config).unwrap();
        assert_eq!(aggregator.name(), "FedAdam");
        let available = AggregatorFactory::available_aggregators();
        assert!(available.contains(&"scaffold"));
        assert!(available.contains(&"fedavgm"));
