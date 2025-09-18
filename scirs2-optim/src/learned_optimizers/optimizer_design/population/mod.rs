//! Population management for evolutionary algorithms
//!
//! This module provides functionality for managing populations of
//! architecture candidates in evolutionary search strategies.

use std::collections::HashMap;

use super::architecture::ArchitectureCandidate;

/// Population manager
pub struct PopulationManager<T: num_traits::Float> {
    /// Current population
    pub population: Vec<ArchitectureCandidate>,
    
    /// Population configuration
    pub config: PopulationConfig,
    
    /// Generation number
    pub generation: usize,
    
    /// Population statistics
    pub stats: PopulationStats<T>,
}

/// Population configuration
#[derive(Debug, Clone)]
pub struct PopulationConfig {
    /// Population size
    pub size: usize,
    
    /// Elite size
    pub elite_size: usize,
    
    /// Diversity maintenance
    pub maintain_diversity: bool,
    
    /// Minimum diversity threshold
    pub min_diversity: f64,
    
    /// Maximum generations
    pub max_generations: usize,
}

/// Population statistics
#[derive(Debug, Clone)]
pub struct PopulationStats<T: num_traits::Float> {
    /// Best fitness in population
    pub best_fitness: T,
    
    /// Average fitness
    pub avg_fitness: T,
    
    /// Worst fitness
    pub worst_fitness: T,
    
    /// Diversity metrics
    pub diversity: DiversityMetrics,
    
    /// Fitness variance
    pub fitness_variance: T,
}

/// Diversity metrics
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    /// Structural diversity
    pub structural: f64,
    
    /// Behavioral diversity
    pub behavioral: f64,
    
    /// Genetic diversity
    pub genetic: f64,
}

impl<T: num_traits::Float + Default + std::fmt::Debug> PopulationManager<T> {
    /// Create new population manager
    pub fn new(config: PopulationConfig) -> Self {
        Self {
            population: Vec::with_capacity(config.size),
            config,
            generation: 0,
            stats: PopulationStats::default(),
        }
    }
    
    /// Initialize random population
    pub fn initialize(&mut self, search_space: &super::space::ArchitectureSearchSpace) {
        self.population.clear();
        
        for i in 0..self.config.size {
            let arch = search_space.sample_random();
            let candidate = ArchitectureCandidate::new(format!("pop_init_{}", i), arch);
            self.population.push(candidate);
        }
        
        self.update_statistics();
    }
    
    /// Update population statistics
    pub fn update_statistics(&mut self) {
        if self.population.is_empty() {
            return;
        }
        
        let fitnesses: Vec<T> = self.population
            .iter()
            .map(|c| num_traits::cast::cast(c.performance.optimization_performance).unwrap_or_else(|| T::zero()))
            .collect();
        
        self.stats.best_fitness = fitnesses.iter().copied().fold(T::neg_infinity(), T::max);
        self.stats.worst_fitness = fitnesses.iter().copied().fold(T::infinity(), T::min);
        
        let sum = fitnesses.iter().fold(T::zero(), |acc, &x| acc + x);
        self.stats.avg_fitness = sum / T::from(fitnesses.len()).unwrap();
        
        // Calculate variance
        let mean = self.stats.avg_fitness;
        let variance_sum = fitnesses.iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(T::zero(), |acc, x| acc + x);
        self.stats.fitness_variance = variance_sum / T::from(fitnesses.len()).unwrap();
    }
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            size: 50,
            elite_size: 5,
            maintain_diversity: true,
            min_diversity: 0.1,
            max_generations: 100,
        }
    }
}

impl<T: num_traits::Float + Default> Default for PopulationStats<T> {
    fn default() -> Self {
        Self {
            best_fitness: T::zero(),
            avg_fitness: T::zero(),
            worst_fitness: T::zero(),
            diversity: DiversityMetrics::default(),
            fitness_variance: T::zero(),
        }
    }
}