//! Progressive Neural Architecture Search
//!
//! This module implements progressive search strategies that gradually increase
//! complexity and search space as the search progresses.

use super::NASConfig;
#[allow(unused_imports)]
use crate::error::Result;
use num_traits::Float;
use std::collections::HashMap;
use std::time::Duration;

/// Progressive NAS configuration
#[derive(Debug, Clone)]
pub struct ProgressiveNAS<T: Float> {
    /// Current search phase
    pub current_phase: SearchPhase,

    /// Phase progression strategy
    pub progression_strategy: ProgressionStrategy,

    /// Complexity scheduler
    pub complexity_scheduler: ComplexityScheduler<T>,

    /// Architecture progression tracker
    pub architecture_progression: ArchitectureProgression<T>,
}

/// Search phases in progressive NAS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SearchPhase {
    /// Initial simple architectures
    Initial,

    /// Intermediate complexity
    Intermediate,

    /// Advanced architectures
    Advanced,

    /// Final optimization phase
    Final,
}

/// Strategy for progressing through phases
#[derive(Debug, Clone)]
pub enum ProgressionStrategy {
    /// Time-based progression
    TimeBased(std::time::Duration),

    /// Performance-based progression
    PerformanceBased(f64),

    /// Budget-based progression
    BudgetBased(usize),

    /// Adaptive progression
    Adaptive,
}

/// Complexity scheduler for progressive search
#[derive(Debug, Clone)]
pub struct ComplexityScheduler<T: Float> {
    /// Current complexity level
    pub current_complexity: T,

    /// Maximum complexity
    pub max_complexity: T,

    /// Complexity increase rate
    pub increase_rate: T,

    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
}

/// Scheduling strategies for complexity increase
#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    /// Linear increase
    Linear,

    /// Exponential increase
    Exponential,

    /// Step-wise increase
    StepWise,

    /// Adaptive based on performance
    Adaptive,
}

/// Architecture progression tracker
#[derive(Debug, Clone)]
pub struct ArchitectureProgression<T: Float> {
    /// Progression history
    pub history: Vec<ProgressionRecord<T>>,

    /// Current best architectures per phase
    pub best_per_phase: HashMap<SearchPhase, Vec<String>>,

    /// Performance trends
    pub performance_trends: Vec<T>,
}

/// Record of progression step
#[derive(Debug, Clone)]
pub struct ProgressionRecord<T: Float> {
    /// Phase when recorded
    pub phase: SearchPhase,

    /// Complexity level
    pub complexity: T,

    /// Best performance achieved
    pub best_performance: T,

    /// Number of architectures evaluated
    pub architectures_evaluated: usize,

    /// Timestamp
    pub timestamp: std::time::Instant,
}

impl<T: Float + Send + Sync + std::iter::Sum> ProgressiveNAS<T> {
    /// Create new progressive NAS
    pub fn new(config: &NASConfig<T>) -> Result<Self> {
        let progression_strategy = match config.search_budget {
            budget if budget < 50 => ProgressionStrategy::TimeBased(Duration::from_secs(300)),
            budget if budget < 200 => ProgressionStrategy::BudgetBased(budget / 4),
            _ => ProgressionStrategy::Adaptive,
        };

        Ok(Self {
            current_phase: SearchPhase::Initial,
            progression_strategy,
            complexity_scheduler: ComplexityScheduler::new()?,
            architecture_progression: ArchitectureProgression::new(),
        })
    }

    /// Update search phase based on progress
    pub fn update_search_phase(&mut self, generation: usize) -> Result<()> {
        match self.progression_strategy {
            ProgressionStrategy::BudgetBased(budget_per_phase) => {
                let phase_index = generation / budget_per_phase;
                self.current_phase = match phase_index {
                    0 => SearchPhase::Initial,
                    1 => SearchPhase::Intermediate,
                    2 => SearchPhase::Advanced,
                    _ => SearchPhase::Final,
                };
            }
            ProgressionStrategy::Adaptive => {
                // Adaptive progression based on performance trends
                if let Some(trend) = self.analyze_performance_trend() {
                    if trend < num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()) && self.current_phase != SearchPhase::Final {
                        self.advance_phase();
                    }
                }
            }
            ProgressionStrategy::TimeBased(duration_per_phase) => {
                let elapsed = self
                    .architecture_progression
                    .history
                    .first()
                    .map(|first| first.timestamp.elapsed())
                    .unwrap_or(Duration::from_secs(0));

                let phases_elapsed = elapsed.as_secs() / duration_per_phase.as_secs();
                self.current_phase = match phases_elapsed {
                    0 => SearchPhase::Initial,
                    1 => SearchPhase::Intermediate,
                    2 => SearchPhase::Advanced,
                    _ => SearchPhase::Final,
                };
            }
            ProgressionStrategy::PerformanceBased(threshold) => {
                if let Some(latest_performance) =
                    self.architecture_progression.performance_trends.last()
                {
                    if *latest_performance > num_traits::cast::cast(threshold).unwrap_or_else(|| T::zero())
                        && self.current_phase != SearchPhase::Final
                    {
                        self.advance_phase();
                    }
                }
            }
        }

        Ok(())
    }

    /// Advance to next phase
    fn advance_phase(&mut self) {
        self.current_phase = match self.current_phase {
            SearchPhase::Initial => SearchPhase::Intermediate,
            SearchPhase::Intermediate => SearchPhase::Advanced,
            SearchPhase::Advanced => SearchPhase::Final,
            SearchPhase::Final => SearchPhase::Final,
        };
    }

    /// Analyze performance trend
    fn analyze_performance_trend(&self) -> Option<T> {
        if self.architecture_progression.performance_trends.len() < 10 {
            return None;
        }

        let recent_trends = &self.architecture_progression.performance_trends
            [self.architecture_progression.performance_trends.len() - 10..];

        // Calculate trend (simplified linear regression slope)
        let n = T::from(recent_trends.len()).unwrap();
        let sum_x = n * (n - T::one()) / num_traits::cast::cast(2).unwrap_or_else(|| T::zero());
        let sum_y = recent_trends.iter().cloned().sum::<T>();
        let sum_xy = recent_trends
            .iter()
            .enumerate()
            .map(|(i, &y)| num_traits::cast::cast(i).unwrap_or_else(|| T::zero()) * y)
            .sum::<T>();
        let sum_x2 = recent_trends
            .iter()
            .enumerate()
            .map(|(i, _)| num_traits::cast::cast(i * i).unwrap_or_else(|| T::zero()))
            .sum::<T>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        Some(slope)
    }

    /// Get current search configuration based on phase
    pub fn get_current_search_config(&self) -> SearchPhaseConfig<T> {
        match self.current_phase {
            SearchPhase::Initial => SearchPhaseConfig {
                complexity_limit: num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()),
                max_components: 2,
                max_depth: 3,
                exploration_factor: num_traits::cast::cast(0.8).unwrap_or_else(|| T::zero()),
                mutation_rate: num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()),
                population_diversity_weight: num_traits::cast::cast(0.7).unwrap_or_else(|| T::zero()),
                conservative_search: true,
            },
            SearchPhase::Intermediate => SearchPhaseConfig {
                complexity_limit: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
                max_components: 4,
                max_depth: 5,
                exploration_factor: num_traits::cast::cast(0.6).unwrap_or_else(|| T::zero()),
                mutation_rate: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
                population_diversity_weight: num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
            SearchPhase::Advanced => SearchPhaseConfig {
                complexity_limit: num_traits::cast::cast(0.75).unwrap_or_else(|| T::zero()),
                max_components: 6,
                max_depth: 7,
                exploration_factor: num_traits::cast::cast(0.4).unwrap_or_else(|| T::zero()),
                mutation_rate: num_traits::cast::cast(0.15).unwrap_or_else(|| T::zero()),
                population_diversity_weight: num_traits::cast::cast(0.3).unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
            SearchPhase::Final => SearchPhaseConfig {
                complexity_limit: T::one(),
                max_components: 8,
                max_depth: 10,
                exploration_factor: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
                mutation_rate: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
                population_diversity_weight: num_traits::cast::cast(0.2).unwrap_or_else(|| T::zero()),
                conservative_search: false,
            },
        }
    }

    /// Record architecture evaluation for progression tracking
    pub fn record_architecture_evaluation(
        &mut self,
        architecture_id: String,
        performance: T,
        complexity: T,
    ) -> Result<()> {
        let record = ProgressionRecord {
            phase: self.current_phase,
            complexity,
            best_performance: performance,
            architectures_evaluated: self.architecture_progression.history.len() + 1,
            timestamp: std::time::Instant::now(),
        };

        self.architecture_progression.record_step(record);

        // Update best architectures for current phase
        let current_phase_key = self.current_phase;
        let best_for_phase = self
            .architecture_progression
            .best_per_phase
            .entry(current_phase_key)
            .or_insert_with(Vec::new);

        // Keep only top 3 architectures per phase
        best_for_phase.push(architecture_id);
        if best_for_phase.len() > 3 {
            best_for_phase.remove(0);
        }

        Ok(())
    }
}

/// Configuration for each search phase
#[derive(Debug, Clone)]
pub struct SearchPhaseConfig<T: Float> {
    /// Maximum complexity allowed in this phase
    pub complexity_limit: T,

    /// Maximum number of components
    pub max_components: usize,

    /// Maximum architecture depth
    pub max_depth: usize,

    /// Exploration factor (0.0 = exploit, 1.0 = explore)
    pub exploration_factor: T,

    /// Mutation rate for genetic algorithms
    pub mutation_rate: T,

    /// Weight for population diversity
    pub population_diversity_weight: T,

    /// Whether to use conservative search strategies
    pub conservative_search: bool,
}

impl<T: Float + Send + Sync> ComplexityScheduler<T> {
    /// Create new complexity scheduler
    pub fn new() -> Result<Self> {
        Ok(Self {
            current_complexity: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            max_complexity: T::one(),
            increase_rate: num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()),
            strategy: SchedulingStrategy::Linear,
        })
    }

    /// Update complexity for current phase
    pub fn update_complexity(&mut self, phase: SearchPhase) -> T {
        match self.strategy {
            SchedulingStrategy::Linear => {
                let phase_factor = match phase {
                    SearchPhase::Initial => num_traits::cast::cast(0.25).unwrap_or_else(|| T::zero()),
                    SearchPhase::Intermediate => num_traits::cast::cast(0.5).unwrap_or_else(|| T::zero()),
                    SearchPhase::Advanced => num_traits::cast::cast(0.75).unwrap_or_else(|| T::zero()),
                    SearchPhase::Final => T::one(),
                };
                self.current_complexity = self.max_complexity * phase_factor;
            }
            SchedulingStrategy::Exponential => {
                self.current_complexity = self.current_complexity * (T::one() + self.increase_rate);
                if self.current_complexity > self.max_complexity {
                    self.current_complexity = self.max_complexity;
                }
            }
            _ => {
                // Other strategies would be implemented here
            }
        }

        self.current_complexity
    }
}

impl<T: Float + Send + Sync> ArchitectureProgression<T> {
    /// Create new architecture progression tracker
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            best_per_phase: HashMap::new(),
            performance_trends: Vec::new(),
        }
    }

    /// Record progression step
    pub fn record_step(&mut self, record: ProgressionRecord<T>) {
        self.performance_trends.push(record.best_performance);
        self.history.push(record);
    }

    /// Get best architectures for current phase
    pub fn get_best_for_phase(&self, phase: SearchPhase) -> Vec<String> {
        self.best_per_phase.get(&phase).cloned().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progressive_nas_creation() {
        let config = NASConfig::<f64>::default();
        let progressive_nas = ProgressiveNAS::new(&config);
        assert!(progressive_nas.is_ok());

        let nas = progressive_nas.unwrap();
        assert_eq!(nas.current_phase, SearchPhase::Initial);
    }

    #[test]
    fn test_complexity_scheduler() {
        let mut scheduler = ComplexityScheduler::<f64>::new().unwrap();

        let initial_complexity = scheduler.update_complexity(SearchPhase::Initial);
        let intermediate_complexity = scheduler.update_complexity(SearchPhase::Intermediate);
        let advanced_complexity = scheduler.update_complexity(SearchPhase::Advanced);

        assert!(initial_complexity < intermediate_complexity);
        assert!(intermediate_complexity < advanced_complexity);
    }

    #[test]
    fn test_architecture_progression() {
        let mut progression = ArchitectureProgression::<f64>::new();

        let record = ProgressionRecord {
            phase: SearchPhase::Initial,
            complexity: 0.1,
            best_performance: 0.8,
            architectures_evaluated: 10,
            timestamp: std::time::Instant::now(),
        };

        progression.record_step(record);
        assert_eq!(progression.history.len(), 1);
        assert_eq!(progression.performance_trends.len(), 1);
    }
}
