//! Convergence detection for optimization processes
//!
//! This module provides sophisticated convergence detection capabilities for optimization
//! algorithms, including statistical tests, ML-based detection, and adaptive thresholds.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use num_traits::Float;

/// Result of convergence analysis
#[derive(Debug, Clone)]
pub struct ConvergenceResult<T: Float> {
    pub converged: bool,
    pub confidence: T,
    pub iterations_to_convergence: Option<usize>,
    pub convergence_rate: Option<T>,
    pub stagnation_count: usize,
    pub trend_analysis: TrendAnalysis<T>,
    pub statistical_significance: T,
}

/// Trend analysis for convergence patterns
#[derive(Debug, Clone)]
pub struct TrendAnalysis<T: Float> {
    pub slope: T,
    pub r_squared: T,
    pub acceleration: T,
    pub volatility: T,
    pub momentum: T,
}

/// Convergence criteria configuration
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria<T: Float> {
    pub absolute_tolerance: T,
    pub relative_tolerance: T,
    pub max_stagnation_iterations: usize,
    pub min_improvement_rate: T,
    pub confidence_threshold: T,
    pub window_size: usize,
    pub statistical_test_threshold: T,
    pub enable_adaptive_thresholds: bool,
    pub early_stopping_patience: usize,
    pub min_iterations: usize,
}

impl<T: Float> Default for ConvergenceCriteria<T> {
    fn default() -> Self {
        Self {
            absolute_tolerance: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
            relative_tolerance: num_traits::cast::cast(1e-4).unwrap_or_else(|| T::zero()),
            max_stagnation_iterations: 50,
            min_improvement_rate: num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()),
            confidence_threshold: num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()),
            window_size: 20,
            statistical_test_threshold: num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()),
            enable_adaptive_thresholds: true,
            early_stopping_patience: 100,
            min_iterations: 10,
        }
    }
}

/// Convergence monitoring state
#[derive(Debug)]
pub struct ConvergenceState<T: Float> {
    pub history: VecDeque<T>,
    pub gradients: VecDeque<T>,
    pub improvement_history: VecDeque<T>,
    pub stagnation_count: usize,
    pub best_value: T,
    pub last_improvement_iteration: usize,
    pub current_iteration: usize,
    pub convergence_start: Option<usize>,
    pub adaptive_tolerance: T,
    pub trend_buffer: VecDeque<T>,
}

impl<T: Float> ConvergenceState<T> {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            gradients: VecDeque::with_capacity(window_size),
            improvement_history: VecDeque::with_capacity(window_size),
            stagnation_count: 0,
            best_value: T::infinity(),
            last_improvement_iteration: 0,
            current_iteration: 0,
            convergence_start: None,
            adaptive_tolerance: num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()),
            trend_buffer: VecDeque::with_capacity(window_size),
        }
    }
}

/// Primary convergence detector
pub struct ConvergenceDetector<T: Float> {
    criteria: ConvergenceCriteria<T>,
    state: ConvergenceState<T>,
    analyzer: ConvergenceAnalyzer<T>,
    monitor: ConvergenceMonitor<T>,
    indicator: ConvergenceIndicator<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> ConvergenceDetector<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        let window_size = criteria.window_size;
        Self {
            analyzer: ConvergenceAnalyzer::new(criteria.clone()),
            monitor: ConvergenceMonitor::new(criteria.clone()),
            indicator: ConvergenceIndicator::new(criteria.clone()),
            state: ConvergenceState::new(window_size),
            criteria,
            _phantom: PhantomData,
        }
    }

    pub fn check_convergence(&mut self, value: T) -> ConvergenceResult<T> {
        self.state.current_iteration += 1;
        self.update_state(value);
        
        // Multi-faceted convergence analysis
        let statistical_result = self.analyzer.statistical_analysis(&self.state);
        let trend_result = self.analyzer.trend_analysis(&self.state);
        let adaptive_result = self.analyzer.adaptive_analysis(&self.state, &self.criteria);
        
        // Combine results with weighted confidence
        let combined_confidence = self.combine_confidences(
            statistical_result.confidence,
            trend_result.confidence,
            adaptive_result.confidence,
        );
        
        let converged = combined_confidence > self.criteria.confidence_threshold
            && self.state.current_iteration >= self.criteria.min_iterations;
        
        if converged && self.state.convergence_start.is_none() {
            self.state.convergence_start = Some(self.state.current_iteration);
        }
        
        let trend_analysis = self.analyzer.compute_trend_analysis(&self.state);
        
        ConvergenceResult {
            converged,
            confidence: combined_confidence,
            iterations_to_convergence: self.state.convergence_start,
            convergence_rate: self.compute_convergence_rate(),
            stagnation_count: self.state.stagnation_count,
            trend_analysis,
            statistical_significance: statistical_result.confidence,
        }
    }

    fn update_state(&mut self, value: T) {
        self.state.history.push_back(value);
        if self.state.history.len() > self.criteria.window_size {
            self.state.history.pop_front();
        }

        // Update best value and improvement tracking
        if value < self.state.best_value {
            let improvement = self.state.best_value - value;
            self.state.improvement_history.push_back(improvement);
            self.state.best_value = value;
            self.state.last_improvement_iteration = self.state.current_iteration;
            self.state.stagnation_count = 0;
        } else {
            self.state.stagnation_count += 1;
            self.state.improvement_history.push_back(T::zero());
        }

        if self.state.improvement_history.len() > self.criteria.window_size {
            self.state.improvement_history.pop_front();
        }

        // Compute gradients if we have enough history
        if self.state.history.len() >= 2 {
            let gradient = self.state.history[self.state.history.len() - 1] 
                         - self.state.history[self.state.history.len() - 2];
            self.state.gradients.push_back(gradient);
            if self.state.gradients.len() > self.criteria.window_size {
                self.state.gradients.pop_front();
            }
        }

        // Update adaptive tolerance
        if self.criteria.enable_adaptive_thresholds {
            self.update_adaptive_tolerance();
        }
    }

    fn update_adaptive_tolerance(&mut self) {
        if self.state.history.len() < 5 {
            return;
        }

        let variance = self.compute_variance(&self.state.history);
        let noise_level = variance.sqrt();
        
        // Adapt tolerance based on noise level and convergence progress
        let progress_factor = if self.state.stagnation_count > 0 {
            T::one() + num_traits::cast::cast(self.state.stagnation_count).unwrap_or_else(|| T::zero()) / num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero())
        } else {
            T::one()
        };

        self.state.adaptive_tolerance = (noise_level * progress_factor)
            .max(self.criteria.absolute_tolerance / num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()))
            .min(self.criteria.absolute_tolerance * num_traits::cast::cast(10.0).unwrap_or_else(|| T::zero()));
    }

    fn compute_variance(&self, data: &VecDeque<T>) -> T {
        if data.len() < 2 {
            return T::zero();
        }

        let mean = data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(data.len()).unwrap();
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(data.len() - 1).unwrap();
        
        variance
    }

    fn combine_confidences(&self, stat_conf: T, trend_conf: T, adapt_conf: T) -> T {
        // Weighted combination of different confidence measures
        let stat_weight = num_traits::cast::cast(0.4).unwrap_or_else(|| T::zero());
        let trend_weight = num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        let adapt_weight = num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        
        stat_conf * stat_weight + trend_conf * trend_weight + adapt_conf * adapt_weight
    }

    fn compute_convergence_rate(&self) -> Option<T> {
        if self.state.improvement_history.len() < 5 {
            return None;
        }

        let recent_improvements: Vec<T> = self.state.improvement_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let avg_improvement = recent_improvements.iter()
            .fold(T::zero(), |acc, &x| acc + x) / T::from(recent_improvements.len()).unwrap();

        Some(avg_improvement)
    }

    pub fn reset(&mut self) {
        self.state = ConvergenceState::new(self.criteria.window_size);
    }

    pub fn get_criteria(&self) -> &ConvergenceCriteria<T> {
        &self.criteria
    }

    pub fn update_criteria(&mut self, criteria: ConvergenceCriteria<T>) {
        self.criteria = criteria;
    }
}

/// Statistical and trend analysis for convergence
pub struct ConvergenceAnalyzer<T: Float> {
    criteria: ConvergenceCriteria<T>,
    _phantom: PhantomData<T>,
}

impl<T: Float> ConvergenceAnalyzer<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            _phantom: PhantomData,
        }
    }

    pub fn statistical_analysis(&self, state: &ConvergenceState<T>) -> ConvergenceResult<T> {
        if state.history.len() < 3 {
            return ConvergenceResult {
                converged: false,
                confidence: T::zero(),
                iterations_to_convergence: None,
                convergence_rate: None,
                stagnation_count: state.stagnation_count,
                trend_analysis: TrendAnalysis {
                    slope: T::zero(),
                    r_squared: T::zero(),
                    acceleration: T::zero(),
                    volatility: T::zero(),
                    momentum: T::zero(),
                },
                statistical_significance: T::zero(),
            };
        }

        // Kolmogorov-Smirnov test for stationarity
        let ks_statistic = self.kolmogorov_smirnov_test(state);
        
        // Mann-Kendall test for trend
        let mk_statistic = self.mann_kendall_test(state);
        
        // Anderson-Darling test for normality of residuals
        let ad_statistic = self.anderson_darling_test(state);
        
        // Combine statistical tests
        let statistical_confidence = self.combine_statistical_tests(ks_statistic, mk_statistic, ad_statistic);
        
        let converged = statistical_confidence > self.criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: statistical_confidence,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis: self.compute_trend_analysis(state),
            statistical_significance: statistical_confidence,
        }
    }

    pub fn trend_analysis(&self, state: &ConvergenceState<T>) -> ConvergenceResult<T> {
        let trend_analysis = self.compute_trend_analysis(state);
        
        // Trend-based convergence assessment
        let trend_confidence = self.assess_trend_convergence(&trend_analysis);
        
        let converged = trend_confidence > self.criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: trend_confidence,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis,
            statistical_significance: trend_confidence,
        }
    }

    pub fn adaptive_analysis(&self, state: &ConvergenceState<T>, criteria: &ConvergenceCriteria<T>) -> ConvergenceResult<T> {
        if state.history.len() < 2 {
            return ConvergenceResult {
                converged: false,
                confidence: T::zero(),
                iterations_to_convergence: None,
                convergence_rate: None,
                stagnation_count: state.stagnation_count,
                trend_analysis: TrendAnalysis {
                    slope: T::zero(),
                    r_squared: T::zero(),
                    acceleration: T::zero(),
                    volatility: T::zero(),
                    momentum: T::zero(),
                },
                statistical_significance: T::zero(),
            };
        }

        let current_value = state.history.back().unwrap();
        let previous_value = state.history[state.history.len() - 2];
        
        // Adaptive tolerance check
        let absolute_improvement = (previous_value - *current_value).abs();
        let relative_improvement = if previous_value.abs() > T::epsilon() {
            absolute_improvement / previous_value.abs()
        } else {
            T::zero()
        };

        let tolerance = if criteria.enable_adaptive_thresholds {
            state.adaptive_tolerance
        } else {
            criteria.absolute_tolerance
        };

        let abs_converged = absolute_improvement < tolerance;
        let rel_converged = relative_improvement < criteria.relative_tolerance;
        let stagnation_converged = state.stagnation_count >= criteria.max_stagnation_iterations;

        let convergence_score = self.compute_adaptive_score(
            abs_converged,
            rel_converged,
            stagnation_converged,
            state,
        );

        let converged = convergence_score > criteria.confidence_threshold;

        ConvergenceResult {
            converged,
            confidence: convergence_score,
            iterations_to_convergence: state.convergence_start,
            convergence_rate: None,
            stagnation_count: state.stagnation_count,
            trend_analysis: self.compute_trend_analysis(state),
            statistical_significance: convergence_score,
        }
    }

    pub fn compute_trend_analysis(&self, state: &ConvergenceState<T>) -> TrendAnalysis<T> {
        if state.history.len() < 3 {
            return TrendAnalysis {
                slope: T::zero(),
                r_squared: T::zero(),
                acceleration: T::zero(),
                volatility: T::zero(),
                momentum: T::zero(),
            };
        }

        let (slope, r_squared) = self.linear_regression(&state.history);
        let acceleration = self.compute_acceleration(&state.history);
        let volatility = self.compute_volatility(&state.history);
        let momentum = self.compute_momentum(&state.history);

        TrendAnalysis {
            slope,
            r_squared,
            acceleration,
            volatility,
            momentum,
        }
    }

    fn kolmogorov_smirnov_test(&self, state: &ConvergenceState<T>) -> T {
        // Simplified KS test for stationarity
        if state.history.len() < 4 {
            return T::zero();
        }

        let n = state.history.len();
        let mid = n / 2;
        
        let first_half: Vec<T> = state.history.iter().take(mid).cloned().collect();
        let second_half: Vec<T> = state.history.iter().skip(mid).cloned().collect();
        
        // Compute empirical CDFs and find maximum difference
        let max_diff = self.compute_cdf_max_difference(&first_half, &second_half);
        
        // Convert to p-value approximation
        let critical_value = num_traits::cast::cast(1.36).unwrap_or_else(|| T::zero()) / (num_traits::cast::cast(n).unwrap_or_else(|| T::zero())).sqrt();
        
        if max_diff < critical_value {
            num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()) // High confidence of stationarity
        } else {
            num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()) // Low confidence
        }
    }

    fn mann_kendall_test(&self, state: &ConvergenceState<T>) -> T {
        // Mann-Kendall test for trend detection
        if state.history.len() < 3 {
            return T::zero();
        }

        let data: Vec<T> = state.history.iter().cloned().collect();
        let n = data.len();
        let mut s = T::zero();

        for i in 0..n-1 {
            for j in i+1..n {
                let diff = data[j] - data[i];
                if diff > T::zero() {
                    s = s + T::one();
                } else if diff < T::zero() {
                    s = s - T::one();
                }
            }
        }

        // Normalize and convert to confidence
        let variance = T::from(n * (n - 1) * (2 * n + 5)).unwrap() / num_traits::cast::cast(18.0).unwrap_or_else(|| T::zero());
        let z = s.abs() / variance.sqrt();
        
        // Convert Z-score to confidence (simplified)
        if z < num_traits::cast::cast(1.96).unwrap_or_else(|| T::zero()) {
            num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()) // No significant trend
        } else {
            num_traits::cast::cast(0.05).unwrap_or_else(|| T::zero()) // Significant trend detected
        }
    }

    fn anderson_darling_test(&self, state: &ConvergenceState<T>) -> T {
        // Simplified Anderson-Darling test for normality
        if state.gradients.len() < 3 {
            return num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero());
        }

        let gradients: Vec<T> = state.gradients.iter().cloned().collect();
        let mean = gradients.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(gradients.len()).unwrap();
        let variance = gradients.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(gradients.len()).unwrap();

        if variance < T::epsilon() {
            return num_traits::cast::cast(0.95).unwrap_or_else(|| T::zero()); // Perfect normality (no variation)
        }

        // Simplified normality assessment based on skewness and kurtosis
        let std_dev = variance.sqrt();
        let skewness = self.compute_skewness(&gradients, mean, std_dev);
        let kurtosis = self.compute_kurtosis(&gradients, mean, std_dev);

        // Convert to confidence score
        let skew_score = (-skewness.abs()).exp();
        let kurt_score = (-(kurtosis - num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero())).abs()).exp();
        
        (skew_score + kurt_score) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
    }

    fn compute_skewness(&self, data: &[T], mean: T, std_dev: T) -> T {
        if std_dev < T::epsilon() {
            return T::zero();
        }

        let n = T::from(data.len()).unwrap();
        let skew = data.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z
            })
            .fold(T::zero(), |acc, x| acc + x) / n;

        skew
    }

    fn compute_kurtosis(&self, data: &[T], mean: T, std_dev: T) -> T {
        if std_dev < T::epsilon() {
            return num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero()); // Normal kurtosis
        }

        let n = T::from(data.len()).unwrap();
        let kurt = data.iter()
            .map(|&x| {
                let z = (x - mean) / std_dev;
                z * z * z * z
            })
            .fold(T::zero(), |acc, x| acc + x) / n;

        kurt
    }

    fn compute_cdf_max_difference(&self, first: &[T], second: &[T]) -> T {
        // Simplified CDF comparison
        let mut first_sorted = first.to_vec();
        let mut second_sorted = second.to_vec();
        first_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        second_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_val = first_sorted[0].min(second_sorted[0]);
        let max_val = first_sorted.last().unwrap().max(*second_sorted.last().unwrap());

        let steps = 20;
        let step_size = (max_val - min_val) / num_traits::cast::cast(steps).unwrap_or_else(|| T::zero());
        let mut max_diff = T::zero();

        for i in 0..=steps {
            let x = min_val + num_traits::cast::cast(i).unwrap_or_else(|| T::zero()) * step_size;
            let cdf1 = self.empirical_cdf(&first_sorted, x);
            let cdf2 = self.empirical_cdf(&second_sorted, x);
            let diff = (cdf1 - cdf2).abs();
            max_diff = max_diff.max(diff);
        }

        max_diff
    }

    fn empirical_cdf(&self, sorted_data: &[T], x: T) -> T {
        let count = sorted_data.iter().filter(|&&val| val <= x).count();
        num_traits::cast::cast(count).unwrap_or_else(|| T::zero()) / T::from(sorted_data.len()).unwrap()
    }

    fn combine_statistical_tests(&self, ks: T, mk: T, ad: T) -> T {
        // Weighted combination of statistical test results
        let ks_weight = num_traits::cast::cast(0.4).unwrap_or_else(|| T::zero());
        let mk_weight = num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        let ad_weight = num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        
        ks * ks_weight + mk * mk_weight + ad * ad_weight
    }

    fn assess_trend_convergence(&self, trend: &TrendAnalysis<T>) -> T {
        // Assess convergence based on trend characteristics
        let slope_score = (-trend.slope.abs()).exp();
        let r_squared_penalty = if trend.r_squared > num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()) {
            num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()) // Strong trend is bad for convergence
        } else {
            T::one()
        };
        let volatility_score = (-trend.volatility).exp();
        let momentum_score = (-trend.momentum.abs()).exp();

        (slope_score * r_squared_penalty + volatility_score + momentum_score) / num_traits::cast::cast(3.0).unwrap_or_else(|| T::zero())
    }

    fn compute_adaptive_score(&self, abs_conv: bool, rel_conv: bool, stag_conv: bool, state: &ConvergenceState<T>) -> T {
        let mut score = T::zero();
        
        if abs_conv {
            score = score + num_traits::cast::cast(0.4).unwrap_or_else(|| T::zero());
        }
        
        if rel_conv {
            score = score + num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        }
        
        if stag_conv {
            score = score + num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero());
        }

        // Adjust score based on improvement history
        if state.improvement_history.len() > 0 {
            let recent_improvements = state.improvement_history.iter()
                .rev()
                .take(5)
                .filter(|&&x| x > T::zero())
                .count();
            
            if recent_improvements == 0 {
                score = score + num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero());
            }
        }

        score.min(T::one())
    }

    fn linear_regression(&self, data: &VecDeque<T>) -> (T, T) {
        if data.len() < 2 {
            return (T::zero(), T::zero());
        }

        let n = T::from(data.len()).unwrap();
        let sum_x = (T::zero()..n).fold(T::zero(), |acc, i| acc + i);
        let sum_y = data.iter().fold(T::zero(), |acc, &y| acc + y);
        let sum_xy = data.iter().enumerate()
            .fold(T::zero(), |acc, (i, &y)| acc + num_traits::cast::cast(i).unwrap_or_else(|| T::zero()) * y);
        let sum_x2 = (T::zero()..n).fold(T::zero(), |acc, i| acc + i * i);

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < T::epsilon() {
            return (T::zero(), T::zero());
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        
        // Compute R²
        let mean_y = sum_y / n;
        let ss_tot = data.iter().fold(T::zero(), |acc, &y| acc + (y - mean_y) * (y - mean_y));
        let ss_res = data.iter().enumerate()
            .fold(T::zero(), |acc, (i, &y)| {
                let predicted = slope * num_traits::cast::cast(i).unwrap_or_else(|| T::zero()) + (sum_y - slope * sum_x) / n;
                acc + (y - predicted) * (y - predicted)
            });

        let r_squared = if ss_tot > T::epsilon() {
            T::one() - ss_res / ss_tot
        } else {
            T::zero()
        };

        (slope, r_squared)
    }

    fn compute_acceleration(&self, data: &VecDeque<T>) -> T {
        if data.len() < 3 {
            return T::zero();
        }

        let mut accelerations = Vec::new();
        for i in 2..data.len() {
            let accel = data[i] - num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * data[i-1] + data[i-2];
            accelerations.push(accel);
        }

        accelerations.iter().fold(T::zero(), |acc, &a| acc + a.abs()) 
            / T::from(accelerations.len()).unwrap()
    }

    fn compute_volatility(&self, data: &VecDeque<T>) -> T {
        if data.len() < 2 {
            return T::zero();
        }

        let mean = data.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(data.len()).unwrap();
        let variance = data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(data.len()).unwrap();
        
        variance.sqrt()
    }

    fn compute_momentum(&self, data: &VecDeque<T>) -> T {
        if data.len() < 3 {
            return T::zero();
        }

        let recent_window = 3.min(data.len());
        let recent_slope = if recent_window >= 2 {
            let start_idx = data.len() - recent_window;
            let end_val = data[data.len() - 1];
            let start_val = data[start_idx];
            (end_val - start_val) / num_traits::cast::cast(recent_window - 1).unwrap_or_else(|| T::zero())
        } else {
            T::zero()
        };

        recent_slope
    }
}

/// Real-time convergence monitoring
pub struct ConvergenceMonitor<T: Float> {
    criteria: ConvergenceCriteria<T>,
    monitoring_interval: Duration,
    last_check: Option<Instant>,
    convergence_history: VecDeque<ConvergenceResult<T>>,
    alerts: Vec<ConvergenceAlert<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct ConvergenceAlert<T: Float> {
    pub timestamp: Instant,
    pub alert_type: ConvergenceAlertType,
    pub message: String,
    pub confidence: T,
    pub suggested_action: String,
}

#[derive(Debug, Clone)]
pub enum ConvergenceAlertType {
    SlowConvergence,
    PrematureConvergence,
    Stagnation,
    Divergence,
    NoiseDetected,
}

impl<T: Float> ConvergenceMonitor<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            monitoring_interval: Duration::from_secs(1),
            last_check: None,
            convergence_history: VecDeque::with_capacity(1000),
            alerts: Vec::new(),
            _phantom: PhantomData,
        }
    }

    pub fn monitor(&mut self, result: ConvergenceResult<T>) {
        let now = Instant::now();
        
        if let Some(last_check) = self.last_check {
            if now.duration_since(last_check) < self.monitoring_interval {
                return;
            }
        }

        self.last_check = Some(now);
        self.convergence_history.push_back(result.clone());
        
        if self.convergence_history.len() > 1000 {
            self.convergence_history.pop_front();
        }

        // Generate alerts based on convergence patterns
        self.check_for_alerts(&result);
    }

    fn check_for_alerts(&mut self, result: &ConvergenceResult<T>) {
        // Check for slow convergence
        if result.stagnation_count > self.criteria.max_stagnation_iterations / 2 {
            self.add_alert(ConvergenceAlertType::Stagnation, 
                "Optimization showing signs of stagnation".to_string(),
                result.confidence);
        }

        // Check for premature convergence
        if result.converged && result.iterations_to_convergence.unwrap_or(1000) < self.criteria.min_iterations {
            self.add_alert(ConvergenceAlertType::PrematureConvergence,
                "Optimization may have converged prematurely".to_string(),
                result.confidence);
        }

        // Check for high volatility
        if result.trend_analysis.volatility > num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()) {
            self.add_alert(ConvergenceAlertType::NoiseDetected,
                "High volatility detected in optimization process".to_string(),
                result.confidence);
        }
    }

    fn add_alert(&mut self, alert_type: ConvergenceAlertType, message: String, confidence: T) {
        let suggested_action = match alert_type {
            ConvergenceAlertType::Stagnation => "Consider adjusting learning rate or optimization strategy".to_string(),
            ConvergenceAlertType::PrematureConvergence => "Verify convergence criteria and consider stricter thresholds".to_string(),
            ConvergenceAlertType::NoiseDetected => "Consider noise reduction or adaptive tolerance".to_string(),
            _ => "Monitor optimization progress closely".to_string(),
        };

        let alert = ConvergenceAlert {
            timestamp: Instant::now(),
            alert_type,
            message,
            confidence,
            suggested_action,
        };

        self.alerts.push(alert);
        
        // Keep only recent alerts
        if self.alerts.len() > 100 {
            self.alerts.remove(0);
        }
    }

    pub fn get_alerts(&self) -> &[ConvergenceAlert<T>] {
        &self.alerts
    }

    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }

    pub fn get_convergence_history(&self) -> &VecDeque<ConvergenceResult<T>> {
        &self.convergence_history
    }
}

/// Convergence indicators and visualization support
pub struct ConvergenceIndicator<T: Float> {
    criteria: ConvergenceCriteria<T>,
    indicator_history: VecDeque<IndicatorValues<T>>,
    _phantom: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct IndicatorValues<T: Float> {
    pub convergence_score: T,
    pub trend_strength: T,
    pub stability_index: T,
    pub confidence_level: T,
    pub progress_ratio: T,
}

impl<T: Float> ConvergenceIndicator<T> {
    pub fn new(criteria: ConvergenceCriteria<T>) -> Self {
        Self {
            criteria,
            indicator_history: VecDeque::with_capacity(1000),
            _phantom: PhantomData,
        }
    }

    pub fn compute_indicators(&mut self, result: &ConvergenceResult<T>, state: &ConvergenceState<T>) -> IndicatorValues<T> {
        let convergence_score = result.confidence;
        let trend_strength = self.compute_trend_strength(&result.trend_analysis);
        let stability_index = self.compute_stability_index(state);
        let confidence_level = result.statistical_significance;
        let progress_ratio = self.compute_progress_ratio(state);

        let indicators = IndicatorValues {
            convergence_score,
            trend_strength,
            stability_index,
            confidence_level,
            progress_ratio,
        };

        self.indicator_history.push_back(indicators.clone());
        if self.indicator_history.len() > 1000 {
            self.indicator_history.pop_front();
        }

        indicators
    }

    fn compute_trend_strength(&self, trend: &TrendAnalysis<T>) -> T {
        // Combine slope and R² to measure trend strength
        let normalized_slope = (-trend.slope.abs()).exp();
        let r_squared_component = trend.r_squared;
        
        (normalized_slope + r_squared_component) / num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero())
    }

    fn compute_stability_index(&self, state: &ConvergenceState<T>) -> T {
        if state.history.len() < 3 {
            return T::zero();
        }

        // Measure stability based on recent volatility
        let recent_window = 10.min(state.history.len());
        let recent_values: Vec<T> = state.history.iter()
            .rev()
            .take(recent_window)
            .cloned()
            .collect();

        let mean = recent_values.iter().fold(T::zero(), |acc, &x| acc + x) / T::from(recent_values.len()).unwrap();
        let variance = recent_values.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x) / T::from(recent_values.len()).unwrap();

        let stability = (-variance.sqrt()).exp();
        stability
    }

    fn compute_progress_ratio(&self, state: &ConvergenceState<T>) -> T {
        if state.history.is_empty() {
            return T::zero();
        }

        let initial_value = state.history[0];
        let current_value = *state.history.back().unwrap();
        let best_value = state.best_value;

        if (initial_value - best_value).abs() < T::epsilon() {
            return T::one();
        }

        let progress = (initial_value - current_value) / (initial_value - best_value);
        progress.max(T::zero()).min(T::one())
    }

    pub fn get_indicator_history(&self) -> &VecDeque<IndicatorValues<T>> {
        &self.indicator_history
    }

    pub fn generate_convergence_report(&self, current_indicators: &IndicatorValues<T>) -> String {
        format!(
            "Convergence Report:\n\
             - Convergence Score: {:.4}\n\
             - Trend Strength: {:.4}\n\
             - Stability Index: {:.4}\n\
             - Confidence Level: {:.4}\n\
             - Progress Ratio: {:.4}\n",
            current_indicators.convergence_score.to_f64().unwrap_or(0.0),
            current_indicators.trend_strength.to_f64().unwrap_or(0.0),
            current_indicators.stability_index.to_f64().unwrap_or(0.0),
            current_indicators.confidence_level.to_f64().unwrap_or(0.0),
            current_indicators.progress_ratio.to_f64().unwrap_or(0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_detector_basic() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let mut detector = ConvergenceDetector::new(criteria);

        // Test with converging sequence
        let values = vec![1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
        
        for value in values {
            let result = detector.check_convergence(value);
            println!("Value: {}, Converged: {}, Confidence: {:.4}", 
                     value, result.converged, result.confidence);
        }
    }

    #[test]
    fn test_statistical_analysis() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);
        
        let mut state = ConvergenceState::new(20);
        let values = vec![10.0, 5.0, 2.5, 1.25, 0.625, 0.3125];
        
        for value in values {
            state.history.push_back(value);
        }
        
        let result = analyzer.statistical_analysis(&state);
        assert!(result.confidence >= 0.0);
        assert!(result.confidence <= 1.0);
    }

    #[test]
    fn test_trend_analysis() {
        let criteria = ConvergenceCriteria::<f64>::default();
        let analyzer = ConvergenceAnalyzer::new(criteria);
        
        let mut state = ConvergenceState::new(20);
        let values = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        
        for value in values {
            state.history.push_back(value);
        }
        
        let trend = analyzer.compute_trend_analysis(&state);
        assert!(trend.slope < 0.0); // Decreasing trend
    }
}