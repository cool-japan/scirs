//! Gradient processing strategies for transformer optimization
//!
//! This module implements various gradient transformation and processing strategies
//! used by the transformer optimizer to improve optimization performance.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque};

use crate::error::{OptimError, Result};

/// Gradient processing strategies
#[derive(Debug, Clone, Copy)]
pub enum GradientProcessingStrategy {
    /// Raw gradients without processing
    Raw,
    /// Gradient clipping
    Clipping,
    /// Gradient normalization
    Normalization,
    /// Adaptive gradient scaling
    AdaptiveScaling,
    /// Gradient smoothing
    Smoothing,
    /// Gradient accumulation
    Accumulation,
    /// Gradient dropout
    Dropout,
    /// Gradient compression
    Compression,
}

/// Gradient processor for transformer optimizer
#[derive(Debug, Clone)]
pub struct GradientProcessor<T: Float> {
    /// Processing strategy
    strategy: GradientProcessingStrategy,
    
    /// Gradient history for smoothing
    gradient_history: VecDeque<Array1<T>>,
    
    /// Accumulated gradients
    accumulated_gradients: Option<Array1<T>>,
    
    /// Gradient statistics
    gradient_stats: GradientStatistics<T>,
    
    /// Processing parameters
    processing_params: GradientProcessingParams<T>,
}

/// Gradient processing parameters
#[derive(Debug, Clone)]
pub struct GradientProcessingParams<T: Float> {
    /// Clipping threshold
    clip_threshold: T,
    
    /// Smoothing factor
    smoothing_factor: T,
    
    /// Accumulation steps
    accumulation_steps: usize,
    
    /// Dropout probability
    dropout_prob: f64,
    
    /// Compression ratio
    compression_ratio: f64,
    
    /// Normalization epsilon
    norm_eps: T,
}

/// Gradient statistics tracking
#[derive(Debug, Clone)]
pub struct GradientStatistics<T: Float> {
    /// Running mean of gradient magnitudes
    mean_magnitude: T,
    
    /// Running variance of gradient magnitudes
    var_magnitude: T,
    
    /// Maximum gradient magnitude seen
    max_magnitude: T,
    
    /// Minimum gradient magnitude seen
    min_magnitude: T,
    
    /// Update count
    update_count: usize,
    
    /// Gradient sparsity
    sparsity: T,
}

impl<T: Float + Default + Clone> GradientProcessor<T> {
    /// Create new gradient processor
    pub fn new(strategy: GradientProcessingStrategy) -> Self {
        Self {
            strategy,
            gradient_history: VecDeque::new(),
            accumulated_gradients: None,
            gradient_stats: GradientStatistics::new(),
            processing_params: GradientProcessingParams::default(),
        }
    }
    
    /// Create with custom parameters
    pub fn new_with_params(
        strategy: GradientProcessingStrategy, 
        params: GradientProcessingParams<T>
    ) -> Self {
        Self {
            strategy,
            gradient_history: VecDeque::new(),
            accumulated_gradients: None,
            gradient_stats: GradientStatistics::new(),
            processing_params: params,
        }
    }

    /// Process gradients according to the selected strategy
    pub fn process_gradients(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        // Update statistics first
        self.gradient_stats.update(gradients);
        
        match self.strategy {
            GradientProcessingStrategy::Raw => Ok(gradients.clone()),
            GradientProcessingStrategy::Clipping => self.clip_gradients(gradients),
            GradientProcessingStrategy::Normalization => self.normalize_gradients(gradients),
            GradientProcessingStrategy::AdaptiveScaling => self.adaptive_scale_gradients(gradients),
            GradientProcessingStrategy::Smoothing => self.smooth_gradients(gradients),
            GradientProcessingStrategy::Accumulation => self.accumulate_gradients(gradients),
            GradientProcessingStrategy::Dropout => self.dropout_gradients(gradients),
            GradientProcessingStrategy::Compression => self.compress_gradients(gradients),
        }
    }

    /// Clip gradients to prevent explosion
    fn clip_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let grad_norm = self.compute_gradient_norm(gradients);
        
        if grad_norm > self.processing_params.clip_threshold {
            let scale = self.processing_params.clip_threshold / grad_norm;
            Ok(gradients * scale)
        } else {
            Ok(gradients.clone())
        }
    }

    /// Normalize gradients
    fn normalize_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let grad_norm = self.compute_gradient_norm(gradients);
        
        if grad_norm > self.processing_params.norm_eps {
            Ok(gradients / grad_norm)
        } else {
            Ok(gradients.clone())
        }
    }

    /// Adaptively scale gradients based on statistics
    fn adaptive_scale_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let current_norm = self.compute_gradient_norm(gradients);
        let mean_norm = self.gradient_stats.mean_magnitude;
        
        if mean_norm > T::zero() {
            let adaptive_scale = num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()) * mean_norm / current_norm + num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
            Ok(gradients * adaptive_scale)
        } else {
            Ok(gradients.clone())
        }
    }

    /// Smooth gradients using exponential moving average
    fn smooth_gradients(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let alpha = self.processing_params.smoothing_factor;
        
        if let Some(prev_grad) = self.gradient_history.back() {
            let smoothed = gradients * alpha + prev_grad * (T::one() - alpha);
            self.gradient_history.push_back(smoothed.clone());
            
            // Keep only recent history
            if self.gradient_history.len() > 10 {
                self.gradient_history.pop_front();
            }
            
            Ok(smoothed)
        } else {
            self.gradient_history.push_back(gradients.clone());
            Ok(gradients.clone())
        }
    }

    /// Accumulate gradients over multiple steps
    fn accumulate_gradients(&mut self, gradients: &Array1<T>) -> Result<Array1<T>> {
        if let Some(ref mut accumulated) = self.accumulated_gradients {
            *accumulated = accumulated.clone() + gradients;
        } else {
            self.accumulated_gradients = Some(gradients.clone());
        }
        
        // Return accumulated gradients if we've reached the target steps
        if self.gradient_stats.update_count % self.processing_params.accumulation_steps == 0 {
            if let Some(accumulated) = self.accumulated_gradients.take() {
                let scale = num_traits::cast::cast(1.0 / self.processing_params.accumulation_steps as f64).unwrap_or_else(|| T::zero());
                Ok(accumulated * scale)
            } else {
                Ok(gradients.clone())
            }
        } else {
            // Return zero gradients for intermediate steps
            Ok(Array1::zeros(gradients.len()))
        }
    }

    /// Apply dropout to gradients
    fn dropout_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        // Simplified dropout - in practice would use proper random sampling
        let mut result = gradients.clone();
        
        // Apply deterministic "dropout" pattern for reproducibility
        for (i, elem) in result.iter_mut().enumerate() {
            if (i % 10) < (self.processing_params.dropout_prob * 10.0) as usize {
                *elem = T::zero();
            }
        }
        
        Ok(result)
    }

    /// Compress gradients (simplified sparsification)
    fn compress_gradients(&self, gradients: &Array1<T>) -> Result<Array1<T>> {
        let mut result = gradients.clone();
        let threshold = self.compute_gradient_norm(gradients) * 
            num_traits::cast::cast(self.processing_params.compression_ratio).unwrap_or_else(|| T::zero());
        
        // Zero out small gradients
        for elem in result.iter_mut() {
            if elem.abs() < threshold {
                *elem = T::zero();
            }
        }
        
        Ok(result)
    }

    /// Compute L2 norm of gradients
    fn compute_gradient_norm(&self, gradients: &Array1<T>) -> T {
        let sum_squares = gradients.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b);
        sum_squares.sqrt()
    }

    /// Get gradient statistics
    pub fn statistics(&self) -> &GradientStatistics<T> {
        &self.gradient_stats
    }

    /// Update processing strategy
    pub fn set_strategy(&mut self, strategy: GradientProcessingStrategy) {
        self.strategy = strategy;
    }

    /// Update processing parameters
    pub fn set_parameters(&mut self, params: GradientProcessingParams<T>) {
        self.processing_params = params;
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.gradient_history.clear();
        self.accumulated_gradients = None;
        self.gradient_stats = GradientStatistics::new();
    }
}

impl<T: Float + Default + Clone> GradientStatistics<T> {
    /// Create new gradient statistics
    pub fn new() -> Self {
        Self {
            mean_magnitude: T::zero(),
            var_magnitude: T::zero(),
            max_magnitude: T::zero(),
            min_magnitude: num_traits::cast::cast(f64::INFINITY).unwrap_or_else(|| T::zero()),
            update_count: 0,
            sparsity: T::zero(),
        }
    }

    /// Update statistics with new gradients
    pub fn update(&mut self, gradients: &Array1<T>) {
        let magnitude = gradients.iter().map(|&x| x * x).fold(T::zero(), |a, b| a + b).sqrt();
        
        self.update_count += 1;
        let count = num_traits::cast::cast(self.update_count as f64).unwrap_or_else(|| T::zero());
        
        // Update running mean
        let delta = magnitude - self.mean_magnitude;
        self.mean_magnitude = self.mean_magnitude + delta / count;
        
        // Update running variance
        let delta2 = magnitude - self.mean_magnitude;
        self.var_magnitude = self.var_magnitude + delta * delta2;
        
        // Update min/max
        if magnitude > self.max_magnitude {
            self.max_magnitude = magnitude;
        }
        if magnitude < self.min_magnitude {
            self.min_magnitude = magnitude;
        }
        
        // Update sparsity (fraction of near-zero elements)
        let zero_count = gradients.iter().filter(|&&x| x.abs() < num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero())).count();
        let current_sparsity = T::from(zero_count as f64 / gradients.len() as f64).unwrap();
        let alpha = num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero());
        self.sparsity = self.sparsity * (T::one() - alpha) + current_sparsity * alpha;
    }

    /// Get mean magnitude
    pub fn mean_magnitude(&self) -> T {
        self.mean_magnitude
    }

    /// Get variance of magnitude
    pub fn variance_magnitude(&self) -> T {
        if self.update_count > 1 {
            self.var_magnitude / T::from((self.update_count - 1) as f64).unwrap()
        } else {
            T::zero()
        }
    }

    /// Get standard deviation of magnitude
    pub fn std_magnitude(&self) -> T {
        self.variance_magnitude().sqrt()
    }

    /// Get gradient sparsity
    pub fn sparsity(&self) -> T {
        self.sparsity
    }
}

impl<T: Float + Default + Clone> Default for GradientProcessingParams<T> {
    fn default() -> Self {
        Self {
            clip_threshold: num_traits::cast::cast(1.0).unwrap_or_else(|| T::zero()),
            smoothing_factor: num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()),
            accumulation_steps: 4,
            dropout_prob: 0.1,
            compression_ratio: 0.1,
            norm_eps: num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()),
        }
    }
}