//! Search history and tracking functionality
//!
//! This module provides functionality for tracking search progress,
//! maintaining search history, and providing analytics.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::architecture::ArchitectureCandidate;

/// Search history manager
pub struct SearchHistory<T: num_traits::Float> {
    /// Search records
    pub records: Vec<SearchRecord<T>>,
    
    /// Best architectures found
    pub best_architectures: Vec<ArchitectureCandidate>,
    
    /// Search statistics
    pub stats: SearchStatistics<T>,
    
    /// Configuration
    pub config: HistoryConfig,
}

/// Search record entry
#[derive(Debug, Clone)]
pub struct SearchRecord<T: num_traits::Float> {
    /// Record ID
    pub id: usize,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Architecture evaluated
    pub architecture: ArchitectureCandidate,
    
    /// Performance achieved
    pub performance: T,
    
    /// Search iteration
    pub iteration: usize,
    
    /// Search strategy used
    pub strategy: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStatistics<T: num_traits::Float> {
    /// Total evaluations
    pub total_evaluations: usize,
    
    /// Total search time
    pub total_time: Duration,
    
    /// Best performance found
    pub best_performance: T,
    
    /// Average performance
    pub avg_performance: T,
    
    /// Performance improvement over time
    pub improvement_rate: T,
    
    /// Convergence metrics
    pub convergence: ConvergenceMetrics<T>,
}

/// Convergence metrics
#[derive(Debug, Clone)]
pub struct ConvergenceMetrics<T: num_traits::Float> {
    /// Has converged
    pub converged: bool,
    
    /// Convergence iteration
    pub convergence_iter: Option<usize>,
    
    /// Stagnation counter
    pub stagnation_count: usize,
    
    /// Last improvement
    pub last_improvement: T,
}

/// History configuration
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    /// Maximum records to keep
    pub max_records: usize,
    
    /// Enable detailed tracking
    pub detailed_tracking: bool,
    
    /// Persistence enabled
    pub persistence: bool,
    
    /// Convergence detection
    pub convergence_detection: ConvergenceConfig,
}

/// Convergence detection configuration
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Enable convergence detection
    pub enabled: bool,
    
    /// Patience (iterations without improvement)
    pub patience: usize,
    
    /// Improvement threshold
    pub threshold: f64,
    
    /// Window size for trend analysis
    pub window_size: usize,
}

impl<T: num_traits::Float + Default + std::fmt::Debug + Clone> SearchHistory<T> {
    /// Create new search history
    pub fn new(config: HistoryConfig) -> Self {
        Self {
            records: Vec::new(),
            best_architectures: Vec::new(),
            stats: SearchStatistics::default(),
            config,
        }
    }
    
    /// Add search record
    pub fn add_record(
        &mut self,
        architecture: ArchitectureCandidate,
        performance: T,
        iteration: usize,
        strategy: String,
    ) {
        let record = SearchRecord {
            id: self.records.len(),
            timestamp: Instant::now(),
            architecture: architecture.clone(),
            performance,
            iteration,
            strategy,
            metadata: HashMap::new(),
        };
        
        self.records.push(record);
        
        // Update best architectures
        if performance > self.stats.best_performance {
            self.stats.best_performance = performance;
            self.best_architectures.insert(0, architecture);
            
            // Keep only top 10
            if self.best_architectures.len() > 10 {
                self.best_architectures.truncate(10);
            }
        }
        
        // Update statistics
        self.update_statistics();
        
        // Check convergence
        if self.config.convergence_detection.enabled {
            self.check_convergence();
        }
        
        // Limit records size
        if self.records.len() > self.config.max_records {
            self.records.remove(0);
        }
    }
    
    /// Update statistics
    fn update_statistics(&mut self) {
        self.stats.total_evaluations = self.records.len();
        
        if !self.records.is_empty() {
            let performances: Vec<T> = self.records.iter().map(|r| r.performance).collect();
            
            let sum = performances.iter().fold(T::zero(), |acc, &x| acc + x);
            self.stats.avg_performance = sum / T::from(performances.len()).unwrap();
            
            // Calculate improvement rate
            if self.records.len() > 1 {
                let recent = self.records[self.records.len() - 1].performance;
                let initial = self.records[0].performance;
                self.stats.improvement_rate = recent - initial;
            }
        }
    }
    
    /// Check for convergence
    fn check_convergence(&mut self) {
        let config = &self.config.convergence_detection;
        
        if self.records.len() < config.window_size {
            return;
        }
        
        // Check recent improvements
        let recent_records = &self.records[self.records.len() - config.window_size..];
        let recent_best = recent_records.iter()
            .map(|r| r.performance)
            .fold(T::neg_infinity(), T::max);
        
        let improvement = recent_best - self.stats.convergence.last_improvement;
        
        if improvement < num_traits::cast::cast(config.threshold).unwrap_or_else(|| T::zero()) {
            self.stats.convergence.stagnation_count += 1;
        } else {
            self.stats.convergence.stagnation_count = 0;
            self.stats.convergence.last_improvement = recent_best;
        }
        
        // Check if converged
        if self.stats.convergence.stagnation_count >= config.patience {
            self.stats.convergence.converged = true;
            self.stats.convergence.convergence_iter = Some(self.records.len());
        }
    }
    
    /// Get search summary
    pub fn get_summary(&self) -> SearchSummary<T> {
        SearchSummary {
            total_evaluations: self.stats.total_evaluations,
            best_performance: self.stats.best_performance,
            avg_performance: self.stats.avg_performance,
            converged: self.stats.convergence.converged,
            convergence_iter: self.stats.convergence.convergence_iter,
        }
    }
}

/// Search summary
#[derive(Debug, Clone)]
pub struct SearchSummary<T: num_traits::Float> {
    pub total_evaluations: usize,
    pub best_performance: T,
    pub avg_performance: T,
    pub converged: bool,
    pub convergence_iter: Option<usize>,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            max_records: 10000,
            detailed_tracking: true,
            persistence: false,
            convergence_detection: ConvergenceConfig {
                enabled: true,
                patience: 50,
                threshold: 0.001,
                window_size: 20,
            },
        }
    }
}

impl<T: num_traits::Float + Default> Default for SearchStatistics<T> {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_time: Duration::from_secs(0),
            best_performance: T::neg_infinity(),
            avg_performance: T::zero(),
            improvement_rate: T::zero(),
            convergence: ConvergenceMetrics {
                converged: false,
                convergence_iter: None,
                stagnation_count: 0,
                last_improvement: T::neg_infinity(),
            },
        }
    }
}