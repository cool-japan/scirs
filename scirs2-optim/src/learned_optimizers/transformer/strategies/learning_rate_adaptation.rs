//! Learning rate adaptation strategies for transformer optimization
//!
//! This module implements various learning rate adaptation strategies that the
//! transformer optimizer can use to dynamically adjust learning rates during training.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::VecDeque;

use crate::error::{OptimError, Result};

/// Learning rate adaptation strategies
#[derive(Debug, Clone, Copy)]
pub enum LearningRateAdaptationStrategy {
    /// Fixed learning rate
    Fixed,
    /// Exponential decay
    ExponentialDecay,
    /// Polynomial decay
    PolynomialDecay,
    /// Cosine annealing
    CosineAnnealing,
    /// Warm restart
    WarmRestart,
    /// Adaptive based on loss
    LossAdaptive,
    /// Adaptive based on gradients
    GradientAdaptive,
    /// Transformer-predicted learning rate
    TransformerPredicted,
}

/// Learning rate adapter for transformer optimizer
#[derive(Debug, Clone)]
pub struct LearningRateAdapter<T: Float> {
    /// Adaptation strategy
    strategy: LearningRateAdaptationStrategy,
    
    /// Base learning rate
    base_lr: T,
    
    /// Current learning rate
    current_lr: T,
    
    /// Adaptation parameters
    adaptation_params: LRAdaptationParams<T>,
    
    /// Loss history for adaptive strategies
    loss_history: VecDeque<T>,
    
    /// Gradient history for adaptive strategies  
    gradient_history: VecDeque<T>,
    
    /// Step counter
    step_count: usize,
    
    /// Epoch counter
    epoch_count: usize,
    
    /// Best loss seen so far
    best_loss: Option<T>,
    
    /// Patience counter for adaptive strategies
    patience_counter: usize,
}

/// Learning rate adaptation parameters
#[derive(Debug, Clone)]
pub struct LRAdaptationParams<T: Float> {
    /// Decay rate for exponential decay
    decay_rate: T,
    
    /// Decay steps for scheduled decay
    decay_steps: usize,
    
    /// Power for polynomial decay
    power: T,
    
    /// Minimum learning rate
    min_lr: T,
    
    /// Maximum learning rate
    max_lr: T,
    
    /// Warmup steps
    warmup_steps: usize,
    
    /// Restart period for warm restart
    restart_period: usize,
    
    /// Patience for loss-based adaptation
    patience: usize,
    
    /// Factor for learning rate reduction
    reduction_factor: T,
    
    /// Threshold for loss improvement
    improvement_threshold: T,
}

/// Learning rate schedule state
#[derive(Debug, Clone)]
pub struct ScheduleState<T: Float> {
    /// Current cycle in warm restart
    current_cycle: usize,
    
    /// Steps in current cycle
    cycle_steps: usize,
    
    /// Whether in warmup phase
    in_warmup: bool,
    
    /// Last schedule update step
    last_update_step: usize,
    
    /// Schedule-specific state
    state: T,
}

impl<T: Float + Default + Clone> LearningRateAdapter<T> {
    /// Create new learning rate adapter
    pub fn new(strategy: LearningRateAdaptationStrategy, base_lr: T) -> Self {
        Self {
            strategy,
            base_lr,
            current_lr: base_lr,
            adaptation_params: LRAdaptationParams::default(),
            loss_history: VecDeque::new(),
            gradient_history: VecDeque::new(),
            step_count: 0,
            epoch_count: 0,
            best_loss: None,
            patience_counter: 0,
        }
    }
    
    /// Create with custom parameters
    pub fn new_with_params(
        strategy: LearningRateAdaptationStrategy,
        base_lr: T,
        params: LRAdaptationParams<T>
    ) -> Self {
        Self {
            strategy,
            base_lr,
            current_lr: base_lr,
            adaptation_params: params,
            loss_history: VecDeque::new(),
            gradient_history: VecDeque::new(),
            step_count: 0,
            epoch_count: 0,
            best_loss: None,
            patience_counter: 0,
        }
    }

    /// Update learning rate based on strategy
    pub fn update_learning_rate(
        &mut self, 
        loss: Option<T>, 
        gradients: Option<&Array1<T>>
    ) -> Result<T> {
        self.step_count += 1;
        
        // Store loss and gradient information
        if let Some(loss_val) = loss {
            self.loss_history.push_back(loss_val);
            if self.loss_history.len() > 100 {
                self.loss_history.pop_front();
            }
        }
        
        if let Some(grad) = gradients {
            let grad_norm = grad.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
            self.gradient_history.push_back(grad_norm);
            if self.gradient_history.len() > 100 {
                self.gradient_history.pop_front();
            }
        }
        
        // Update learning rate based on strategy
        self.current_lr = match self.strategy {
            LearningRateAdaptationStrategy::Fixed => self.base_lr,
            LearningRateAdaptationStrategy::ExponentialDecay => self.exponential_decay(),
            LearningRateAdaptationStrategy::PolynomialDecay => self.polynomial_decay(),
            LearningRateAdaptationStrategy::CosineAnnealing => self.cosine_annealing(),
            LearningRateAdaptationStrategy::WarmRestart => self.warm_restart(),
            LearningRateAdaptationStrategy::LossAdaptive => self.loss_adaptive(loss)?,
            LearningRateAdaptationStrategy::GradientAdaptive => self.gradient_adaptive(gradients)?,
            LearningRateAdaptationStrategy::TransformerPredicted => self.transformer_predicted()?,
        };
        
        // Apply warmup if in warmup phase
        if self.step_count < self.adaptation_params.warmup_steps {
            let warmup_factor = num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero()) / 
                num_traits::cast::cast(self.adaptation_params.warmup_steps as f64).unwrap_or_else(|| T::zero());
            self.current_lr = self.current_lr * warmup_factor;
        }
        
        // Clamp to min/max bounds
        self.current_lr = self.current_lr.max(self.adaptation_params.min_lr)
            .min(self.adaptation_params.max_lr);
        
        Ok(self.current_lr)
    }

    /// Exponential decay schedule
    fn exponential_decay(&self) -> T {
        let steps = num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero());
        let decay_steps = num_traits::cast::cast(self.adaptation_params.decay_steps as f64).unwrap_or_else(|| T::zero());
        let decay_factor = (steps / decay_steps) * self.adaptation_params.decay_rate.ln();
        self.base_lr * (-decay_factor).exp()
    }

    /// Polynomial decay schedule
    fn polynomial_decay(&self) -> T {
        if self.step_count >= self.adaptation_params.decay_steps {
            self.adaptation_params.min_lr
        } else {
            let progress = num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero()) / 
                num_traits::cast::cast(self.adaptation_params.decay_steps as f64).unwrap_or_else(|| T::zero());
            let decay_factor = (T::one() - progress).powf(self.adaptation_params.power);
            (self.base_lr - self.adaptation_params.min_lr) * decay_factor + 
                self.adaptation_params.min_lr
        }
    }

    /// Cosine annealing schedule
    fn cosine_annealing(&self) -> T {
        let steps = num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero());
        let total_steps = num_traits::cast::cast(self.adaptation_params.decay_steps as f64).unwrap_or_else(|| T::zero());
        let pi = num_traits::cast::cast(std::f64::consts::PI).unwrap_or_else(|| T::zero());
        
        let cosine_factor = (T::one() + (pi * steps / total_steps).cos()) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
        self.adaptation_params.min_lr + 
            (self.base_lr - self.adaptation_params.min_lr) * cosine_factor
    }

    /// Warm restart schedule
    fn warm_restart(&self) -> T {
        let period = self.adaptation_params.restart_period;
        let cycle_position = self.step_count % period;
        let progress = num_traits::cast::cast(cycle_position as f64).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(period as f64).unwrap_or_else(|| T::zero());
        
        let pi = num_traits::cast::cast(std::f64::consts::PI).unwrap_or_else(|| T::zero());
        let cosine_factor = (T::one() + (pi * progress).cos()) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero());
        
        self.adaptation_params.min_lr + 
            (self.base_lr - self.adaptation_params.min_lr) * cosine_factor
    }

    /// Loss-adaptive learning rate adjustment
    fn loss_adaptive(&mut self, loss: Option<T>) -> Result<T> {
        if let Some(current_loss) = loss {
            if let Some(best_loss) = self.best_loss {
                let improvement = (best_loss - current_loss) / best_loss;
                
                if improvement > self.adaptation_params.improvement_threshold {
                    // Loss improved significantly, reset patience and potentially increase LR
                    self.best_loss = Some(current_loss);
                    self.patience_counter = 0;
                    
                    // Slight increase in learning rate if loss is improving well
                    self.current_lr * num_traits::cast::cast(1.01).unwrap_or_else(|| T::zero())
                } else {
                    // Loss didn't improve sufficiently
                    self.patience_counter += 1;
                    
                    if self.patience_counter >= self.adaptation_params.patience {
                        // Reduce learning rate
                        self.patience_counter = 0;
                        self.current_lr * self.adaptation_params.reduction_factor
                    } else {
                        self.current_lr
                    }
                }
            } else {
                // First loss value
                self.best_loss = Some(current_loss);
                self.current_lr
            }
        } else {
            self.current_lr
        }
    }

    /// Gradient-adaptive learning rate adjustment
    fn gradient_adaptive(&self, gradients: Option<&Array1<T>>) -> Result<T> {
        if let Some(grad) = gradients {
            let grad_norm = grad.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
            
            // Adaptive learning rate based on gradient magnitude
            let target_norm = num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero());
            let scale_factor = target_norm / (grad_norm + num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()));
            
            // Smooth the adaptation
            let alpha = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
            let adapted_lr = self.current_lr * (T::one() - alpha) + 
                (self.base_lr * scale_factor) * alpha;
            
            adapted_lr
        } else {
            Ok(self.current_lr)
        }
    }

    /// Transformer-predicted learning rate (placeholder)
    fn transformer_predicted(&self) -> Result<T> {
        // This would use a separate transformer network to predict optimal LR
        // For now, use a simple heuristic based on step count
        let decay_factor = T::one() / (T::one() + num_traits::cast::cast(self.step_count as f64).unwrap_or_else(|| T::zero()) * num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()));
        Ok(self.base_lr * decay_factor)
    }

    /// Get current learning rate
    pub fn current_learning_rate(&self) -> T {
        self.current_lr
    }

    /// Get base learning rate
    pub fn base_learning_rate(&self) -> T {
        self.base_lr
    }

    /// Set base learning rate
    pub fn set_base_learning_rate(&mut self, lr: T) {
        self.base_lr = lr;
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Mark epoch end
    pub fn on_epoch_end(&mut self) {
        self.epoch_count += 1;
    }

    /// Reset adapter state
    pub fn reset(&mut self) {
        self.step_count = 0;
        self.epoch_count = 0;
        self.current_lr = self.base_lr;
        self.loss_history.clear();
        self.gradient_history.clear();
        self.best_loss = None;
        self.patience_counter = 0;
    }

    /// Get loss history
    pub fn loss_history(&self) -> &VecDeque<T> {
        &self.loss_history
    }

    /// Get gradient history
    pub fn gradient_history(&self) -> &VecDeque<T> {
        &self.gradient_history
    }

    /// Update strategy
    pub fn set_strategy(&mut self, strategy: LearningRateAdaptationStrategy) {
        self.strategy = strategy;
    }

    /// Update parameters
    pub fn set_parameters(&mut self, params: LRAdaptationParams<T>) {
        self.adaptation_params = params;
    }
}

impl<T: Float + Default + Clone> Default for LRAdaptationParams<T> {
    fn default() -> Self {
        Self {
            decay_rate: num_traits::cast::cast(0.96).unwrap_or_else(|| T::zero()),
            decay_steps: 1000,
            power: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            min_lr: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
            max_lr: num_traits::cast::cast(1e-1).unwrap_or_else(|| T::zero()),
            warmup_steps: 100,
            restart_period: 1000,
            patience: 10,
            reduction_factor: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
            improvement_threshold: num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()),
        }
    }
}