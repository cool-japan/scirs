//! Regularization strategies specific to transformer optimization
//!
//! This module implements transformer-specific regularization techniques that
//! work in conjunction with the attention mechanisms and optimization dynamics.

#![allow(dead_code)]

use ndarray::{Array1, Array2, Array3};
use num_traits::Float;
use std::collections::HashMap;

use crate::error::{OptimError, Result};

/// Regularization strategies for transformer optimization
#[derive(Debug, Clone, Copy)]
pub enum RegularizationStrategy {
    /// No regularization
    None,
    /// Standard L2 weight decay
    L2WeightDecay,
    /// L1 sparsity regularization
    L1Sparsity,
    /// Attention entropy regularization
    AttentionEntropy,
    /// Gradient penalty regularization
    GradientPenalty,
    /// Spectral normalization
    SpectralNorm,
    /// Attention diversity regularization
    AttentionDiversity,
    /// Parameter orthogonality constraints
    Orthogonality,
    /// Adaptive regularization based on training dynamics
    Adaptive,
}

/// Transformer regularizer
#[derive(Debug, Clone)]
pub struct TransformerRegularizer<T: Float> {
    /// Regularization strategy
    strategy: RegularizationStrategy,
    
    /// Regularization parameters
    regularization_params: RegularizationParams<T>,
    
    /// Attention pattern history for entropy calculations
    attention_history: Vec<Array3<T>>,
    
    /// Parameter statistics for adaptive regularization
    parameter_stats: HashMap<String, ParameterStatistics<T>>,
    
    /// Spectral normalization state
    spectral_state: Option<SpectralNormState<T>>,
    
    /// Step counter for adaptive strategies
    step_count: usize,
}

/// Regularization parameters
#[derive(Debug, Clone)]
pub struct RegularizationParams<T: Float> {
    /// L2 regularization coefficient
    l2_weight: T,
    
    /// L1 regularization coefficient
    l1_weight: T,
    
    /// Attention entropy regularization weight
    entropy_weight: T,
    
    /// Gradient penalty coefficient
    gradient_penalty_weight: T,
    
    /// Spectral normalization power iterations
    spectral_iterations: usize,
    
    /// Attention diversity weight
    diversity_weight: T,
    
    /// Orthogonality constraint weight
    orthogonality_weight: T,
    
    /// Adaptive regularization base strength
    adaptive_base_strength: T,
    
    /// Adaptive regularization decay rate
    adaptive_decay_rate: T,
}

/// Parameter statistics for adaptive regularization
#[derive(Debug, Clone)]
pub struct ParameterStatistics<T: Float> {
    /// Running mean of parameter magnitudes
    mean_magnitude: T,
    
    /// Running variance of parameter magnitudes
    var_magnitude: T,
    
    /// Parameter update frequency
    update_frequency: T,
    
    /// Gradient-to-parameter ratio
    grad_param_ratio: T,
    
    /// Update count
    update_count: usize,
}

/// State for spectral normalization
#[derive(Debug, Clone)]
pub struct SpectralNormState<T: Float> {
    /// Left singular vectors for each weight matrix
    u_vectors: HashMap<String, Array1<T>>,
    
    /// Right singular vectors for each weight matrix
    v_vectors: HashMap<String, Array1<T>>,
    
    /// Spectral norms for each weight matrix
    spectral_norms: HashMap<String, T>,
}

impl<T: Float + Default + Clone> TransformerRegularizer<T> {
    /// Create new transformer regularizer
    pub fn new(strategy: RegularizationStrategy) -> Self {
        Self {
            strategy,
            regularization_params: RegularizationParams::default(),
            attention_history: Vec::new(),
            parameter_stats: HashMap::new(),
            spectral_state: None,
            step_count: 0,
        }
    }
    
    /// Create with custom parameters
    pub fn new_with_params(
        strategy: RegularizationStrategy,
        params: RegularizationParams<T>
    ) -> Self {
        Self {
            strategy,
            regularization_params: params,
            attention_history: Vec::new(),
            parameter_stats: HashMap::new(),
            spectral_state: None,
            step_count: 0,
        }
    }

    /// Apply regularization to parameters and gradients
    pub fn apply_regularization(
        &mut self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>
    ) -> Result<T> {
        self.step_count += 1;
        
        // Store attention patterns for entropy regularization
        if let Some(attention) = attention_patterns {
            self.attention_history.push(attention.clone());
            // Keep only recent history
            if self.attention_history.len() > 100 {
                self.attention_history.remove(0);
            }
        }
        
        match self.strategy {
            RegularizationStrategy::None => Ok(T::zero()),
            RegularizationStrategy::L2WeightDecay => self.apply_l2_regularization(parameters, gradients),
            RegularizationStrategy::L1Sparsity => self.apply_l1_regularization(parameters, gradients),
            RegularizationStrategy::AttentionEntropy => self.apply_attention_entropy_regularization(gradients, attention_patterns),
            RegularizationStrategy::GradientPenalty => self.apply_gradient_penalty(gradients),
            RegularizationStrategy::SpectralNorm => self.apply_spectral_normalization(parameters, gradients),
            RegularizationStrategy::AttentionDiversity => self.apply_attention_diversity_regularization(gradients, attention_patterns),
            RegularizationStrategy::Orthogonality => self.apply_orthogonality_regularization(parameters, gradients),
            RegularizationStrategy::Adaptive => self.apply_adaptive_regularization(parameters, gradients),
        }
    }

    /// Apply L2 weight decay regularization
    fn apply_l2_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();
        
        for (param_name, param_values) in parameters {
            if let Some(grad) = gradients.get_mut(param_name) {
                // Add L2 penalty to gradients
                let l2_grad = param_values * self.regularization_params.l2_weight;
                *grad = grad.clone() + &l2_grad;
                
                // Compute L2 loss contribution
                let l2_loss = param_values.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss + l2_loss * self.regularization_params.l2_weight * num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
            }
        }
        
        Ok(total_reg_loss)
    }

    /// Apply L1 sparsity regularization
    fn apply_l1_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();
        
        for (param_name, param_values) in parameters {
            if let Some(grad) = gradients.get_mut(param_name) {
                // Add L1 penalty to gradients
                let l1_grad = param_values.mapv(|x| if x > T::zero() { 
                    self.regularization_params.l1_weight 
                } else { 
                    -self.regularization_params.l1_weight 
                });
                *grad = grad.clone() + &l1_grad;
                
                // Compute L1 loss contribution
                let l1_loss = param_values.iter().map(|&x| x.abs()).fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss + l1_loss * self.regularization_params.l1_weight;
            }
        }
        
        Ok(total_reg_loss)
    }

    /// Apply attention entropy regularization
    fn apply_attention_entropy_regularization(
        &self,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>
    ) -> Result<T> {
        if let Some(attention) = attention_patterns {
            let entropy_penalty = self.compute_attention_entropy(attention)?;
            
            // Apply entropy penalty to attention-related gradients
            if let Some(attention_grad) = gradients.get_mut("attention_weights") {
                let entropy_grad = self.compute_entropy_gradient(attention)?;
                *attention_grad = attention_grad.clone() + &entropy_grad;
            }
            
            Ok(entropy_penalty * self.regularization_params.entropy_weight)
        } else {
            Ok(T::zero())
        }
    }

    /// Apply gradient penalty regularization
    fn apply_gradient_penalty(
        &self,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        let mut total_penalty = T::zero();
        
        for (_param_name, grad) in gradients {
            // Compute gradient penalty (penalize large gradients)
            let grad_norm_squared = grad.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
            let penalty_grad = grad * (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * self.regularization_params.gradient_penalty_weight);
            *grad = grad.clone() + &penalty_grad;
            
            total_penalty = total_penalty + grad_norm_squared * self.regularization_params.gradient_penalty_weight;
        }
        
        Ok(total_penalty)
    }

    /// Apply spectral normalization
    fn apply_spectral_normalization(
        &mut self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        if self.spectral_state.is_none() {
            self.spectral_state = Some(SpectralNormState {
                u_vectors: HashMap::new(),
                v_vectors: HashMap::new(),
                spectral_norms: HashMap::new(),
            });
        }
        
        let mut total_reg_loss = T::zero();
        
        if let Some(ref mut spectral_state) = self.spectral_state {
            for (param_name, param_values) in parameters {
                // Compute spectral norm and apply normalization
                let spectral_norm = self.compute_spectral_norm(param_name, param_values, spectral_state)?;
                
                if spectral_norm > T::one() {
                    // Apply spectral normalization to gradients
                    if let Some(grad) = gradients.get_mut(param_name) {
                        let normalization_factor = T::one() / spectral_norm;
                        *grad = grad.clone() * normalization_factor;
                        
                        // Add regularization loss
                        total_reg_loss = total_reg_loss + (spectral_norm - T::one()).powi(2);
                    }
                }
            }
        }
        
        Ok(total_reg_loss)
    }

    /// Apply attention diversity regularization
    fn apply_attention_diversity_regularization(
        &self,
        gradients: &mut HashMap<String, Array2<T>>,
        attention_patterns: Option<&Array3<T>>
    ) -> Result<T> {
        if let Some(attention) = attention_patterns {
            let diversity_penalty = self.compute_attention_diversity(attention)?;
            
            // Apply diversity penalty to attention gradients
            if let Some(attention_grad) = gradients.get_mut("attention_weights") {
                let diversity_grad = self.compute_diversity_gradient(attention)?;
                *attention_grad = attention_grad.clone() + &diversity_grad;
            }
            
            Ok(diversity_penalty * self.regularization_params.diversity_weight)
        } else {
            Ok(T::zero())
        }
    }

    /// Apply orthogonality regularization
    fn apply_orthogonality_regularization(
        &self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();
        
        for (param_name, param_values) in parameters {
            // Apply orthogonality constraint to weight matrices
            if param_name.contains("weight") && param_values.nrows() == param_values.ncols() {
                let orthogonality_penalty = self.compute_orthogonality_penalty(param_values)?;
                
                if let Some(grad) = gradients.get_mut(param_name) {
                    let ortho_grad = self.compute_orthogonality_gradient(param_values)?;
                    *grad = grad.clone() + &ortho_grad;
                }
                
                total_reg_loss = total_reg_loss + orthogonality_penalty * self.regularization_params.orthogonality_weight;
            }
        }
        
        Ok(total_reg_loss)
    }

    /// Apply adaptive regularization based on training dynamics
    fn apply_adaptive_regularization(
        &mut self,
        parameters: &HashMap<String, Array2<T>>,
        gradients: &mut HashMap<String, Array2<T>>
    ) -> Result<T> {
        let mut total_reg_loss = T::zero();
        
        for (param_name, param_values) in parameters {
            // Update parameter statistics
            self.update_parameter_statistics(param_name, param_values, gradients.get(param_name));
            
            // Compute adaptive regularization strength
            let adaptive_strength = self.compute_adaptive_strength(param_name)?;
            
            if let Some(grad) = gradients.get_mut(param_name) {
                // Apply adaptive L2 regularization
                let adaptive_grad = param_values * adaptive_strength;
                *grad = grad.clone() + &adaptive_grad;
                
                // Compute regularization loss
                let reg_loss = param_values.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
                total_reg_loss = total_reg_loss + reg_loss * adaptive_strength * num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
            }
        }
        
        Ok(total_reg_loss)
    }

    /// Compute attention entropy
    fn compute_attention_entropy(&self, attention: &Array3<T>) -> Result<T> {
        let (num_heads, seq_len, _) = attention.dim();
        let mut total_entropy = T::zero();
        
        for h in 0..num_heads {
            for i in 0..seq_len {
                let mut entropy = T::zero();
                for j in 0..seq_len {
                    let p = attention[[h, i, j]];
                    if p > T::zero() {
                        entropy = entropy - p * p.ln();
                    }
                }
                total_entropy = total_entropy + entropy;
            }
        }
        
        Ok(total_entropy / T::from((num_heads * seq_len) as f64).unwrap())
    }

    /// Compute entropy gradient (simplified)
    fn compute_entropy_gradient(&self, attention: &Array3<T>) -> Result<Array2<T>> {
        let (num_heads, seq_len) = (attention.shape()[0], attention.shape()[1]);
        let mut grad = Array2::zeros((num_heads, seq_len));
        
        for h in 0..num_heads {
            for i in 0..seq_len {
                // Simplified entropy gradient
                grad[[h, i]] = -self.regularization_params.entropy_weight;
            }
        }
        
        Ok(grad)
    }

    /// Compute attention diversity penalty
    fn compute_attention_diversity(&self, attention: &Array3<T>) -> Result<T> {
        let (num_heads, seq_len, _) = attention.dim();
        let mut diversity_penalty = T::zero();
        
        // Penalize similarity between attention heads
        for h1 in 0..num_heads {
            for h2 in (h1 + 1)..num_heads {
                let mut similarity = T::zero();
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        similarity = similarity + attention[[h1, i, j]] * attention[[h2, i, j]];
                    }
                }
                diversity_penalty = diversity_penalty + similarity * similarity;
            }
        }
        
        Ok(diversity_penalty)
    }

    /// Compute diversity gradient (simplified)
    fn compute_diversity_gradient(&self, attention: &Array3<T>) -> Result<Array2<T>> {
        let (num_heads, seq_len) = (attention.shape()[0], attention.shape()[1]);
        let mut grad = Array2::zeros((num_heads, seq_len));
        
        // Simplified diversity gradient computation
        for h in 0..num_heads {
            for i in 0..seq_len {
                grad[[h, i]] = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()) * self.regularization_params.diversity_weight;
            }
        }
        
        Ok(grad)
    }

    /// Compute orthogonality penalty for square matrices
    fn compute_orthogonality_penalty(&self, matrix: &Array2<T>) -> Result<T> {
        if matrix.nrows() != matrix.ncols() {
            return Ok(T::zero());
        }
        
        let n = matrix.nrows();
        let mut penalty = T::zero();
        
        // Compute W^T W - I and measure its Frobenius norm
        for i in 0..n {
            for j in 0..n {
                let mut dot_product = T::zero();
                for k in 0..n {
                    dot_product = dot_product + matrix[[k, i]] * matrix[[k, j]];
                }
                
                let expected = if i == j { T::one() } else { T::zero() };
                let diff = dot_product - expected;
                penalty = penalty + diff * diff;
            }
        }
        
        Ok(penalty)
    }

    /// Compute orthogonality gradient
    fn compute_orthogonality_gradient(&self, matrix: &Array2<T>) -> Result<Array2<T>> {
        if matrix.nrows() != matrix.ncols() {
            return Ok(Array2::zeros(matrix.dim()));
        }
        
        let n = matrix.nrows();
        let mut grad = Array2::zeros((n, n));
        
        // Gradient of ||W^T W - I||^2 with respect to W
        for i in 0..n {
            for j in 0..n {
                let mut grad_val = T::zero();
                for k in 0..n {
                    let wtw_ik = (0..n).map(|l| matrix[[l, i]] * matrix[[l, k]]).fold(T::zero(), |a, b| a + b);
                    let expected = if i == k { T::one() } else { T::zero() };
                    grad_val = grad_val + num_traits::cast::cast(4.0).unwrap_or_else(|| T::zero()) * (wtw_ik - expected) * matrix[[j, k]];
                }
                grad[[j, i]] = grad_val * self.regularization_params.orthogonality_weight;
            }
        }
        
        Ok(grad)
    }

    /// Update parameter statistics for adaptive regularization
    fn update_parameter_statistics(
        &mut self,
        param_name: &str,
        parameters: &Array2<T>,
        gradients: Option<&Array2<T>>
    ) {
        let param_magnitude = parameters.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
        
        let stats = self.parameter_stats.entry(param_name.to_string())
            .or_insert(ParameterStatistics::new());
        
        stats.update_count += 1;
        let alpha = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
        
        // Update running mean
        stats.mean_magnitude = stats.mean_magnitude * (T::one() - alpha) + param_magnitude * alpha;
        
        // Update gradient-to-parameter ratio if gradients available
        if let Some(grad) = gradients {
            let grad_magnitude = grad.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
            if param_magnitude > T::zero() {
                let ratio = grad_magnitude / param_magnitude;
                stats.grad_param_ratio = stats.grad_param_ratio * (T::one() - alpha) + ratio * alpha;
            }
        }
    }

    /// Compute adaptive regularization strength
    fn compute_adaptive_strength(&self, param_name: &str) -> Result<T> {
        if let Some(stats) = self.parameter_stats.get(param_name) {
            let base_strength = self.regularization_params.adaptive_base_strength;
            let decay_rate = self.regularization_params.adaptive_decay_rate;
            
            // Adaptive strength based on parameter statistics
            let magnitude_factor = T::one() / (T::one() + stats.mean_magnitude);
            let step_decay = decay_rate.powf(num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero()));
            
            Ok(base_strength * magnitude_factor * step_decay)
        } else {
            Ok(self.regularization_params.adaptive_base_strength)
        }
    }

    /// Compute spectral norm using power iteration
    fn compute_spectral_norm(
        &self,
        param_name: &str,
        matrix: &Array2<T>,
        spectral_state: &mut SpectralNormState<T>
    ) -> Result<T> {
        let (m, n) = matrix.dim();
        
        // Initialize u and v vectors if not present
        if !spectral_state.u_vectors.contains_key(param_name) {
            spectral_state.u_vectors.insert(param_name.to_string(), Array1::ones(m));
            spectral_state.v_vectors.insert(param_name.to_string(), Array1::ones(n));
        }
        
        // Power iteration to find largest singular value
        let mut u = spectral_state.u_vectors[param_name].clone();
        let mut v = spectral_state.v_vectors[param_name].clone();
        
        for _ in 0..self.regularization_params.spectral_iterations {
            // v = W^T u / ||W^T u||
            let mut new_v = Array1::zeros(n);
            for j in 0..n {
                for i in 0..m {
                    new_v[j] = new_v[j] + matrix[[i, j]] * u[i];
                }
            }
            let v_norm = new_v.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
            if v_norm > T::zero() {
                v = new_v / v_norm;
            }
            
            // u = W v / ||W v||
            let mut new_u = Array1::zeros(m);
            for i in 0..m {
                for j in 0..n {
                    new_u[i] = new_u[i] + matrix[[i, j]] * v[j];
                }
            }
            let u_norm = new_u.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
            if u_norm > T::zero() {
                u = new_u / u_norm;
            }
        }
        
        // Compute spectral norm: u^T W v
        let mut spectral_norm = T::zero();
        for i in 0..m {
            for j in 0..n {
                spectral_norm = spectral_norm + u[i] * matrix[[i, j]] * v[j];
            }
        }
        
        // Update state
        spectral_state.u_vectors.insert(param_name.to_string(), u);
        spectral_state.v_vectors.insert(param_name.to_string(), v);
        spectral_state.spectral_norms.insert(param_name.to_string(), spectral_norm);
        
        Ok(spectral_norm)
    }

    /// Get regularization statistics
    pub fn get_statistics(&self) -> HashMap<String, T> {
        let mut stats = HashMap::new();
        
        stats.insert("step_count".to_string(), num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero()));
        stats.insert("attention_history_length".to_string(), T::from(self.attention_history.len() as f64).unwrap());
        
        if let Some(ref spectral_state) = self.spectral_state {
            for (param_name, &norm) in &spectral_state.spectral_norms {
                stats.insert(format!("spectral_norm_{}", param_name), norm);
            }
        }
        
        stats
    }

    /// Reset regularizer state
    pub fn reset(&mut self) {
        self.attention_history.clear();
        self.parameter_stats.clear();
        self.spectral_state = None;
        self.step_count = 0;
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: RegularizationStrategy) {
        self.strategy = strategy;
    }

    /// Update parameters
    pub fn set_parameters(&mut self, params: RegularizationParams<T>) {
        self.regularization_params = params;
    }
}

impl<T: Float + Default + Clone> ParameterStatistics<T> {
    fn new() -> Self {
        Self {
            mean_magnitude: T::zero(),
            var_magnitude: T::zero(),
            update_frequency: T::zero(),
            grad_param_ratio: T::zero(),
            update_count: 0,
        }
    }
}

impl<T: Float + Default + Clone> Default for RegularizationParams<T> {
    fn default() -> Self {
        Self {
            l2_weight: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            l1_weight: num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()),
            entropy_weight: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            gradient_penalty_weight: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            spectral_iterations: 1,
            diversity_weight: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            orthogonality_weight: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            adaptive_base_strength: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
            adaptive_decay_rate: num_traits::cast::cast(0.99).unwrap_or_else(|| T::zero()),
        }
    }
}