//! Population management for evolutionary neural architecture search

use num_traits::Float;
use rand::seq::SliceRandom;
use std::collections::{HashMap, VecDeque};
use std::cmp::Ordering;
use crate::error::Result;
use super::evaluation::EvaluationMetrics;

/// Population manager for evolutionary algorithms
pub struct PopulationManager<T: Float> {
    /// Current population
    population: Vec<Individual<T>>,
    /// Population size
    population_size: usize,
    /// Elite size (number of best individuals to keep)
    elite_size: usize,
    /// Selection strategy
    selection_strategy: SelectionStrategy,
    /// Population history
    history: VecDeque<PopulationSnapshot<T>>,
    /// Diversity tracker
    diversity_tracker: DiversityTracker,
    /// Generation counter
    generation: usize,
}

impl<T: Float> PopulationManager<T> {
    /// Create new population manager
    pub fn new(population_size: usize, elite_size: usize) -> Result<Self> {
        if elite_size > population_size {
            return Err(crate::error::OptimError::Other(
                "Elite size cannot be larger than population size".to_string()
            ));
        }

        Ok(Self {
            population: Vec::new(),
            population_size,
            elite_size,
            selection_strategy: SelectionStrategy::Tournament,
            history: VecDeque::new(),
            diversity_tracker: DiversityTracker::new(),
            generation: 0,
        })
    }

    /// Initialize population with architectures and their evaluations
    pub fn initialize(
        &mut self,
        architectures: Vec<String>,
        evaluations: Vec<EvaluationMetrics>,
    ) -> Result<()> {
        if architectures.len() != evaluations.len() {
            return Err(crate::error::OptimError::Other(
                "Architecture and evaluation counts must match".to_string()
            ));
        }

        self.population.clear();

        for (architecture, metrics) in architectures.into_iter().zip(evaluations.into_iter()) {
            let individual = Individual::new(architecture, metrics)?;
            self.population.push(individual);
        }

        // Sort by fitness
        self.sort_population_by_fitness();

        // Record initial population
        self.record_population_snapshot();

        // Update diversity
        self.diversity_tracker.update(&self.population);

        Ok(())
    }

    /// Update population with new individuals
    pub fn update(
        &mut self,
        new_architectures: Vec<String>,
        new_evaluations: Vec<EvaluationMetrics>,
    ) -> Result<()> {
        if new_architectures.len() != new_evaluations.len() {
            return Err(crate::error::OptimError::Other(
                "Architecture and evaluation counts must match".to_string()
            ));
        }

        // Create new individuals
        let mut new_individuals = Vec::new();
        for (architecture, metrics) in new_architectures.into_iter().zip(new_evaluations.into_iter()) {
            let individual = Individual::new(architecture, metrics)?;
            new_individuals.push(individual);
        }

        // Add new individuals to population
        self.population.extend(new_individuals);

        // Sort by fitness
        self.sort_population_by_fitness();

        // Select survivors
        self.select_survivors()?;

        // Update generation
        self.generation += 1;

        // Record population state
        self.record_population_snapshot();

        // Update diversity
        self.diversity_tracker.update(&self.population);

        Ok(())
    }

    /// Sort population by fitness (descending order)
    fn sort_population_by_fitness(&mut self) {
        self.population.sort_by(|a, b| {
            b.fitness.partial_cmp(&a.fitness).unwrap_or(Ordering::Equal)
        });
    }

    /// Select survivors for next generation
    fn select_survivors(&mut self) -> Result<()> {
        match self.selection_strategy {
            SelectionStrategy::Elite => self.elite_selection(),
            SelectionStrategy::Tournament => self.tournament_selection(),
            SelectionStrategy::Roulette => self.roulette_selection(),
            SelectionStrategy::Rank => self.rank_selection(),
            SelectionStrategy::NSGA2 => self.nsga2_selection(),
        }
    }

    /// Elite selection (keep best individuals)
    fn elite_selection(&mut self) -> Result<()> {
        self.population.truncate(self.population_size);
        Ok(())
    }

    /// Tournament selection
    fn tournament_selection(&mut self) -> Result<()> {
        let tournament_size = 3;
        let mut survivors = Vec::new();

        // Keep elite individuals
        let elite_count = self.elite_size.min(self.population.len());
        survivors.extend_from_slice(&self.population[..elite_count]);

        // Fill remaining slots with tournament selection
        while survivors.len() < self.population_size && !self.population.is_empty() {
            let winner = self.run_tournament(tournament_size)?;
            survivors.push(winner);
        }

        self.population = survivors;
        Ok(())
    }

    /// Run tournament selection
    fn run_tournament(&self, tournament_size: usize) -> Result<Individual<T>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let tournament_size = tournament_size.min(self.population.len());
        let mut tournament_indices: Vec<usize> = (0..self.population.len()).collect();
        tournament_indices.shuffle(&mut rng);
        let tournament: Vec<_> = tournament_indices
            .into_iter()
            .take(tournament_size)
            .map(|i| &self.population[i])
            .collect();

        tournament
            .into_iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Ordering::Equal))
            .cloned()
            .ok_or_else(|| crate::error::OptimError::Other("Tournament failed".to_string()))
    }

    /// Roulette wheel selection
    fn roulette_selection(&mut self) -> Result<()> {
        // Convert fitness to selection probabilities
        let total_fitness: T = self.population.iter().map(|ind| ind.fitness).fold(T::zero(), |acc, f| acc + f);

        if total_fitness <= T::zero() {
            return self.elite_selection(); // Fallback
        }

        let mut survivors = Vec::new();
        let mut rng = rand::thread_rng();

        // Keep elite individuals
        let elite_count = self.elite_size.min(self.population.len());
        survivors.extend_from_slice(&self.population[..elite_count]);

        // Select remaining individuals
        while survivors.len() < self.population_size {
            let random_value = T::from(rand::random::<f64>()).unwrap() * total_fitness;
            let mut cumulative_fitness = T::zero();

            for individual in &self.population {
                cumulative_fitness = cumulative_fitness + individual.fitness;
                if cumulative_fitness >= random_value {
                    survivors.push(individual.clone());
                    break;
                }
            }
        }

        self.population = survivors;
        Ok(())
    }

    /// Rank-based selection
    fn rank_selection(&mut self) -> Result<()> {
        // Already sorted by fitness
        let mut survivors = Vec::new();

        // Keep elite individuals
        let elite_count = self.elite_size.min(self.population.len());
        survivors.extend_from_slice(&self.population[..elite_count]);

        // Select remaining based on rank probabilities
        let remaining_slots = self.population_size - elite_count;
        let total_ranks: f64 = (self.population.len() * (self.population.len() + 1) / 2) as f64;

        for _ in 0..remaining_slots {
            let random_value = rand::random::<f64>() * total_ranks;
            let mut cumulative_rank = 0.0;

            for (rank, individual) in self.population.iter().enumerate() {
                cumulative_rank += (self.population.len() - rank) as f64;
                if cumulative_rank >= random_value {
                    survivors.push(individual.clone());
                    break;
                }
            }
        }

        self.population = survivors;
        Ok(())
    }

    /// NSGA-II selection (simplified multi-objective)
    fn nsga2_selection(&mut self) -> Result<()> {
        // Simplified NSGA-II implementation
        // In practice, this would handle multiple objectives properly
        self.elite_selection()
    }

    /// Get current population
    pub fn get_current_population(&self) -> &[Individual<T>] {
        &self.population
    }

    /// Get best individual
    pub fn get_best_individual(&self) -> Option<&Individual<T>> {
        self.population.first()
    }

    /// Get best performance
    pub fn get_best_performance(&self) -> Option<T> {
        self.get_best_individual().map(|ind| ind.fitness)
    }

    /// Get top architectures
    pub fn get_top_architectures(&self, count: usize) -> Vec<String> {
        self.population
            .iter()
            .take(count)
            .map(|ind| ind.architecture.clone())
            .collect()
    }

    /// Record population snapshot for history
    fn record_population_snapshot(&mut self) {
        let snapshot = PopulationSnapshot {
            generation: self.generation,
            best_fitness: self.get_best_performance().unwrap_or(T::zero()),
            average_fitness: self.calculate_average_fitness(),
            population_size: self.population.len(),
            diversity_score: self.diversity_tracker.get_current_diversity(),
        };

        self.history.push_back(snapshot);

        // Limit history size
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    /// Calculate average fitness
    fn calculate_average_fitness(&self) -> T {
        if self.population.is_empty() {
            return T::zero();
        }

        let total: T = self.population.iter().map(|ind| ind.fitness).fold(T::zero(), |acc, f| acc + f);
        total / T::from(self.population.len()).unwrap()
    }

    /// Get population statistics
    pub fn get_statistics(&self) -> PopulationStatistics<T> {
        PopulationStatistics {
            generation: self.generation,
            population_size: self.population.len(),
            best_fitness: self.get_best_performance().unwrap_or(T::zero()),
            average_fitness: self.calculate_average_fitness(),
            worst_fitness: self.population.last().map(|ind| ind.fitness).unwrap_or(T::zero()),
            diversity_score: self.diversity_tracker.get_current_diversity(),
            convergence_history: self.get_convergence_history(),
        }
    }

    /// Get convergence history
    fn get_convergence_history(&self) -> Vec<T> {
        self.history.iter().map(|snapshot| snapshot.best_fitness).collect()
    }

    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: SelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Check if population has converged
    pub fn has_converged(&self, threshold: T, window_size: usize) -> bool {
        if self.history.len() < window_size {
            return false;
        }

        let recent_history: Vec<_> = self.history
            .iter()
            .rev()
            .take(window_size)
            .map(|s| s.best_fitness)
            .collect();

        let max_fitness = recent_history.iter().fold(T::neg_infinity(), |acc, &f| acc.max(f));
        let min_fitness = recent_history.iter().fold(T::infinity(), |acc, &f| acc.min(f));

        (max_fitness - min_fitness) < threshold
    }
}

/// Individual in the population
#[derive(Debug, Clone)]
pub struct Individual<T: Float> {
    /// Architecture representation
    pub architecture: String,
    /// Evaluation metrics
    pub metrics: EvaluationMetrics,
    /// Fitness value (derived from metrics)
    pub fitness: T,
    /// Age (number of generations survived)
    pub age: usize,
    /// Unique identifier
    pub id: String,
}

impl<T: Float> Individual<T> {
    /// Create new individual
    pub fn new(architecture: String, metrics: EvaluationMetrics) -> Result<Self> {
        let fitness = T::from(metrics.overall_score(&[])).unwrap();
        let id = format!("ind_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());

        Ok(Self {
            architecture,
            metrics,
            fitness,
            age: 0,
            id,
        })
    }

    /// Update fitness based on new metrics
    pub fn update_fitness(&mut self, weights: &[f64]) {
        self.fitness = T::from(self.metrics.overall_score(weights)).unwrap();
    }

    /// Increment age
    pub fn increment_age(&mut self) {
        self.age += 1;
    }
}

/// Selection strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionStrategy {
    /// Elite selection (keep best)
    Elite,
    /// Tournament selection
    Tournament,
    /// Roulette wheel selection
    Roulette,
    /// Rank-based selection
    Rank,
    /// NSGA-II for multi-objective
    NSGA2,
}

/// Population snapshot for history tracking
#[derive(Debug, Clone)]
pub struct PopulationSnapshot<T: Float> {
    pub generation: usize,
    pub best_fitness: T,
    pub average_fitness: T,
    pub population_size: usize,
    pub diversity_score: f64,
}

/// Population statistics
#[derive(Debug, Clone)]
pub struct PopulationStatistics<T: Float> {
    pub generation: usize,
    pub population_size: usize,
    pub best_fitness: T,
    pub average_fitness: T,
    pub worst_fitness: T,
    pub diversity_score: f64,
    pub convergence_history: Vec<T>,
}

/// Diversity tracker
#[derive(Debug)]
pub struct DiversityTracker {
    current_diversity: f64,
    diversity_history: VecDeque<f64>,
}

impl DiversityTracker {
    pub fn new() -> Self {
        Self {
            current_diversity: 0.0,
            diversity_history: VecDeque::new(),
        }
    }

    pub fn update<T: Float>(&mut self, population: &[Individual<T>]) {
        self.current_diversity = self.calculate_diversity(population);
        self.diversity_history.push_back(self.current_diversity);

        if self.diversity_history.len() > 100 {
            self.diversity_history.pop_front();
        }
    }

    fn calculate_diversity<T: Float>(&self, population: &[Individual<T>]) -> f64 {
        if population.len() < 2 {
            return 0.0;
        }

        // Calculate diversity based on architecture uniqueness
        let mut unique_architectures = std::collections::HashSet::new();
        for individual in population {
            unique_architectures.insert(individual.architecture.clone());
        }

        unique_architectures.len() as f64 / population.len() as f64
    }

    pub fn get_current_diversity(&self) -> f64 {
        self.current_diversity
    }

    pub fn get_diversity_trend(&self) -> DiversityTrend {
        if self.diversity_history.len() < 2 {
            return DiversityTrend::Stable;
        }

        let recent = self.diversity_history.back().unwrap();
        let previous = self.diversity_history[self.diversity_history.len() - 2];

        if recent > &(previous + 0.05) {
            DiversityTrend::Increasing
        } else if recent < &(previous - 0.05) {
            DiversityTrend::Decreasing
        } else {
            DiversityTrend::Stable
        }
    }
}

/// Diversity trend
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiversityTrend {
    Increasing,
    Decreasing,
    Stable,
}

// UUID placeholder implementation
mod uuid {
    pub struct Uuid;

    impl Uuid {
        pub fn new_v4() -> Self { Self }

        pub fn to_string(&self) -> String {
            format!("xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics(accuracy: f64) -> EvaluationMetrics {
        EvaluationMetrics {
            accuracy,
            training_time_seconds: 100.0,
            inference_time_ms: 10.0,
            memory_usage_mb: 512,
            flops: 1_000_000,
            parameters: 100_000,
            energy_consumption: 50.0,
            convergence_rate: 0.8,
            robustness_score: 0.7,
            generalization_score: accuracy * 0.9,
            efficiency_score: 0.6,
            valid: true,
        }
    }

    #[test]
    fn test_population_manager_creation() {
        let manager = PopulationManager::<f32>::new(10, 3);
        assert!(manager.is_ok());

        let invalid_manager = PopulationManager::<f32>::new(10, 15);
        assert!(invalid_manager.is_err());
    }

    #[test]
    fn test_population_initialization() {
        let mut manager = PopulationManager::<f32>::new(5, 2).unwrap();

        let architectures = vec![
            "arch1".to_string(),
            "arch2".to_string(),
            "arch3".to_string(),
        ];

        let evaluations = vec![
            create_test_metrics(0.8),
            create_test_metrics(0.9),
            create_test_metrics(0.7),
        ];

        let result = manager.initialize(architectures, evaluations);
        assert!(result.is_ok());

        assert_eq!(manager.population.len(), 3);
        assert_eq!(manager.get_best_performance().unwrap(), 0.9f32);
    }

    #[test]
    fn test_individual_creation() {
        let metrics = create_test_metrics(0.85);
        let individual = Individual::<f32>::new("test_arch".to_string(), metrics);
        assert!(individual.is_ok());

        let ind = individual.unwrap();
        assert_eq!(ind.architecture, "test_arch");
        assert!(ind.fitness > 0.0);
    }

    #[test]
    fn test_diversity_tracker() {
        let mut tracker = DiversityTracker::new();

        let individuals = vec![
            Individual::<f64>::new("arch1".to_string(), create_test_metrics(0.8)).unwrap(),
            Individual::<f64>::new("arch2".to_string(), create_test_metrics(0.9)).unwrap(),
            Individual::<f64>::new("arch1".to_string(), create_test_metrics(0.7)).unwrap(), // Duplicate
        ];

        tracker.update(&individuals);
        assert!(tracker.current_diversity > 0.0);
        assert!(tracker.current_diversity < 1.0); // Due to duplicate
    }
}