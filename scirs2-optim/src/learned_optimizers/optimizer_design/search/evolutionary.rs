//! Evolutionary search strategies for neural architecture search
//!
//! This module implements evolutionary algorithms for NAS, including genetic algorithms,
//! evolution strategies, and other population-based optimization methods.

use std::collections::HashMap;
use num_traits::Float;

use super::super::architecture::{ArchitectureSpec, ArchitectureCandidate};
use super::strategies::{MutationOperator, CrossoverOperator, SelectionMethod};

/// Evolutionary search state
#[derive(Debug)]
pub struct EvolutionarySearchState<T: Float> {
    /// Current population
    pub population: Vec<ArchitectureCandidate>,

    /// Generation number
    pub generation: usize,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,

    /// Fitness history
    pub fitness_history: Vec<Vec<f64>>,

    /// Selection pressure
    pub selection_pressure: f64,

    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,

    /// Evolutionary operators
    pub mutation_operators: Vec<Box<dyn MutationOperator>>,
    pub crossover_operators: Vec<Box<dyn CrossoverOperator>>,
    pub selection_methods: Vec<Box<dyn SelectionMethod<T>>>,

    /// Evolutionary parameters
    pub parameters: EvolutionaryParameters,
}

/// Population diversity metrics
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    /// Structural diversity
    pub structural_diversity: f64,

    /// Performance diversity
    pub performance_diversity: f64,

    /// Genotypic diversity
    pub genotypic_diversity: f64,

    /// Phenotypic diversity
    pub phenotypic_diversity: f64,
}

/// Evolutionary algorithm parameters
#[derive(Debug, Clone)]
pub struct EvolutionaryParameters {
    /// Population size
    pub population_size: usize,

    /// Elite size (number of best individuals to keep)
    pub elite_size: usize,

    /// Mutation rate
    pub mutation_rate: f64,

    /// Crossover rate
    pub crossover_rate: f64,

    /// Selection pressure
    pub selection_pressure: f64,

    /// Maximum generations
    pub max_generations: usize,

    /// Diversity maintenance
    pub maintain_diversity: bool,

    /// Minimum diversity threshold
    pub min_diversity_threshold: f64,

    /// Fitness sharing enabled
    pub fitness_sharing: bool,

    /// Speciation enabled
    pub speciation: bool,

    /// Multi-objective optimization
    pub multi_objective: bool,
}

impl Default for EvolutionaryParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            elite_size: 5,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            selection_pressure: 2.0,
            max_generations: 100,
            maintain_diversity: true,
            min_diversity_threshold: 0.1,
            fitness_sharing: false,
            speciation: false,
            multi_objective: false,
        }
    }
}

impl<T: Float + Default + std::fmt::Debug> EvolutionarySearchState<T> {
    /// Create new evolutionary search state
    pub fn new(population_size: usize) -> Self {
        Self {
            population: Vec::with_capacity(population_size),
            generation: 0,
            _phantom: std::marker::PhantomData,
            fitness_history: Vec::new(),
            selection_pressure: 2.0,
            diversity_metrics: DiversityMetrics::default(),
            mutation_operators: Vec::new(),
            crossover_operators: Vec::new(),
            selection_methods: Vec::new(),
            parameters: EvolutionaryParameters::default(),
        }
    }

    /// Initialize population with random architectures
    pub fn initialize_population(&mut self, search_space: &super::super::space::ArchitectureSearchSpace) {
        self.population.clear();
        
        for i in 0..self.parameters.population_size {
            let arch_spec = self.generate_random_architecture(search_space, i);
            let candidate = ArchitectureCandidate::new(
                format!("gen0_ind{}", i),
                arch_spec,
            );
            self.population.push(candidate);
        }
    }

    /// Generate random architecture from search space
    fn generate_random_architecture(
        &self,
        _search_space: &super::super::space::ArchitectureSearchSpace,
        seed: usize,
    ) -> ArchitectureSpec {
        // Simplified random architecture generation
        use super::super::architecture::{LayerSpec, LayerDimensions, LayerType, ActivationType, GlobalArchitectureConfig};
        
        let num_layers = 1 + (seed % 5); // 1-5 layers
        let mut layers = Vec::new();

        for i in 0..num_layers {
            let layer_type = match i % 3 {
                0 => LayerType::Linear,
                1 => LayerType::LSTM,
                _ => LayerType::Attention,
            };

            let dimensions = LayerDimensions {
                input_dim: if i == 0 { 128 } else { 64 },
                output_dim: 64,
                hidden_dims: vec![],
            };

            let activation = match seed % 3 {
                0 => ActivationType::ReLU,
                1 => ActivationType::GELU,
                _ => ActivationType::Tanh,
            };

            layers.push(LayerSpec::new(layer_type, dimensions, activation));
        }

        ArchitectureSpec::new(layers, GlobalArchitectureConfig::default())
    }

    /// Evolve population for one generation
    pub fn evolve_generation(&mut self) -> Result<(), super::SearchError> {
        // Calculate fitness for all individuals
        let fitnesses = self.calculate_population_fitness();

        // Update diversity metrics
        self.update_diversity_metrics();

        // Record fitness history
        self.fitness_history.push(fitnesses.clone());

        // Create new population
        let mut new_population = Vec::new();

        // Elite selection - keep best individuals
        let elite_indices = self.select_elite(&fitnesses);
        for &idx in &elite_indices {
            new_population.push(self.population[idx].clone());
        }

        // Generate offspring to fill rest of population
        while new_population.len() < self.parameters.population_size {
            if rand::random::<f64>() < self.parameters.crossover_rate {
                // Crossover
                let offspring = self.generate_crossover_offspring(&fitnesses)?;
                new_population.extend(offspring);
            } else {
                // Mutation only
                let offspring = self.generate_mutation_offspring(&fitnesses)?;
                new_population.push(offspring);
            }
        }

        // Truncate if we generated too many offspring
        new_population.truncate(self.parameters.population_size);

        // Replace population
        self.population = new_population;
        self.generation += 1;

        Ok(())
    }

    /// Calculate fitness for all individuals in population
    fn calculate_population_fitness(&self) -> Vec<f64> {
        self.population
            .iter()
            .map(|individual| individual.performance.optimization_performance)
            .collect()
    }

    /// Select elite individuals
    fn select_elite(&self, fitnesses: &[f64]) -> Vec<usize> {
        let mut indexed_fitnesses: Vec<(usize, f64)> = fitnesses
            .iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();

        indexed_fitnesses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_fitnesses
            .into_iter()
            .take(self.parameters.elite_size)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Generate offspring through crossover
    fn generate_crossover_offspring(&self, fitnesses: &[f64]) -> Result<Vec<ArchitectureCandidate>, super::SearchError> {
        // Select parents
        let parent1_idx = self.select_parent(fitnesses)?;
        let parent2_idx = self.select_parent(fitnesses)?;

        if parent1_idx == parent2_idx {
            // If same parent selected, just mutate
            return Ok(vec![self.generate_mutation_offspring(fitnesses)?]);
        }

        let parent1 = &self.population[parent1_idx].architecture;
        let parent2 = &self.population[parent2_idx].architecture;

        // Apply crossover (use first crossover operator if available)
        if let Some(crossover_op) = self.crossover_operators.first() {
            let (child1_arch, child2_arch) = crossover_op.crossover(parent1, parent2)?;

            let mut offspring = Vec::new();

            // Create candidates from offspring architectures
            let child1 = ArchitectureCandidate::new(
                format!("gen{}_cross1", self.generation + 1),
                child1_arch,
            );
            offspring.push(child1);

            let child2 = ArchitectureCandidate::new(
                format!("gen{}_cross2", self.generation + 1),
                child2_arch,
            );
            offspring.push(child2);

            // Apply mutation to offspring if enabled
            for child in &mut offspring {
                if rand::random::<f64>() < self.parameters.mutation_rate {
                    self.apply_mutations(&mut child.architecture)?;
                }
            }

            Ok(offspring)
        } else {
            // No crossover operator available, fall back to mutation
            Ok(vec![self.generate_mutation_offspring(fitnesses)?])
        }
    }

    /// Generate offspring through mutation
    fn generate_mutation_offspring(&self, fitnesses: &[f64]) -> Result<ArchitectureCandidate, super::SearchError> {
        let parent_idx = self.select_parent(fitnesses)?;
        let mut child_arch = self.population[parent_idx].architecture.clone();

        // Apply mutations
        self.apply_mutations(&mut child_arch)?;

        let child = ArchitectureCandidate::new(
            format!("gen{}_mut", self.generation + 1),
            child_arch,
        );

        Ok(child)
    }

    /// Select parent for reproduction
    fn select_parent(&self, fitnesses: &[f64]) -> Result<usize, super::SearchError> {
        if let Some(selection_method) = self.selection_methods.first() {
            let selected = selection_method.select(&self.population, &fitnesses.iter().map(|&f| num_traits::cast::cast(f).unwrap_or_else(|| T::zero())).collect::<Vec<_>>(), 1);
            selected.first().copied().ok_or_else(|| {
                super::SearchError::GenerationFailed("No parent selected".to_string())
            })
        } else {
            // Default to random selection
            Ok(rand::random::<usize>() % self.population.len())
        }
    }

    /// Apply mutations to architecture
    fn apply_mutations(&self, architecture: &mut ArchitectureSpec) -> Result<(), super::SearchError> {
        for mutation_op in &self.mutation_operators {
            if rand::random::<f64>() < self.parameters.mutation_rate {
                mutation_op.mutate(architecture, self.parameters.mutation_rate);
            }
        }
        Ok(())
    }

    /// Update diversity metrics
    fn update_diversity_metrics(&mut self) {
        self.diversity_metrics = calculate_diversity_metrics(&self.population);
    }

    /// Check if evolution should terminate
    pub fn should_terminate(&self) -> bool {
        // Terminate if max generations reached
        if self.generation >= self.parameters.max_generations {
            return true;
        }

        // Terminate if diversity too low (premature convergence)
        if self.parameters.maintain_diversity
            && self.diversity_metrics.structural_diversity < self.parameters.min_diversity_threshold
        {
            return true;
        }

        // Terminate if no improvement for many generations
        if self.fitness_history.len() >= 20 {
            let recent_best = self.fitness_history
                .iter()
                .rev()
                .take(10)
                .filter_map(|gen| gen.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let older_best = self.fitness_history
                .iter()
                .rev()
                .skip(10)
                .take(10)
                .filter_map(|gen| gen.iter().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if let (Some(&recent), Some(&older)) = (recent_best, older_best) {
                if (recent - older).abs() < 0.001 {
                    return true; // No significant improvement
                }
            }
        }

        false
    }

    /// Get best individuals in current population
    pub fn get_best_individuals(&self, n: usize) -> Vec<&ArchitectureCandidate> {
        let mut indexed_individuals: Vec<(usize, &ArchitectureCandidate)> = self
            .population
            .iter()
            .enumerate()
            .collect();

        indexed_individuals.sort_by(|a, b| {
            b.1.performance
                .optimization_performance
                .partial_cmp(&a.1.performance.optimization_performance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed_individuals
            .into_iter()
            .take(n)
            .map(|(_, individual)| individual)
            .collect()
    }

    /// Add mutation operator
    pub fn add_mutation_operator(&mut self, operator: Box<dyn MutationOperator>) {
        self.mutation_operators.push(operator);
    }

    /// Add crossover operator
    pub fn add_crossover_operator(&mut self, operator: Box<dyn CrossoverOperator>) {
        self.crossover_operators.push(operator);
    }

    /// Add selection method
    pub fn add_selection_method(&mut self, method: Box<dyn SelectionMethod<T>>) {
        self.selection_methods.push(method);
    }

    /// Set parameters
    pub fn set_parameters(&mut self, parameters: EvolutionaryParameters) {
        self.parameters = parameters;
    }
}

/// Calculate diversity metrics for a population
pub fn calculate_diversity_metrics(population: &[ArchitectureCandidate]) -> DiversityMetrics {
    if population.len() < 2 {
        return DiversityMetrics::default();
    }

    let structural_diversity = calculate_structural_diversity(population);
    let performance_diversity = calculate_performance_diversity(population);
    let genotypic_diversity = calculate_genotypic_diversity(population);
    let phenotypic_diversity = calculate_phenotypic_diversity(population);

    DiversityMetrics {
        structural_diversity,
        performance_diversity,
        genotypic_diversity,
        phenotypic_diversity,
    }
}

fn calculate_structural_diversity(population: &[ArchitectureCandidate]) -> f64 {
    let mut total_distance = 0.0;
    let mut comparisons = 0;

    for i in 0..population.len() {
        for j in (i + 1)..population.len() {
            let distance = structural_distance(&population[i].architecture, &population[j].architecture);
            total_distance += distance;
            comparisons += 1;
        }
    }

    if comparisons > 0 {
        total_distance / comparisons as f64
    } else {
        0.0
    }
}

fn calculate_performance_diversity(population: &[ArchitectureCandidate]) -> f64 {
    let performances: Vec<f64> = population
        .iter()
        .map(|ind| ind.performance.optimization_performance)
        .collect();

    if performances.len() < 2 {
        return 0.0;
    }

    let mean = performances.iter().sum::<f64>() / performances.len() as f64;
    let variance = performances
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / performances.len() as f64;

    variance.sqrt()
}

fn calculate_genotypic_diversity(population: &[ArchitectureCandidate]) -> f64 {
    // Simplified genotypic diversity based on architecture encoding differences
    calculate_structural_diversity(population)
}

fn calculate_phenotypic_diversity(population: &[ArchitectureCandidate]) -> f64 {
    // Phenotypic diversity based on performance characteristics
    calculate_performance_diversity(population)
}

fn structural_distance(arch1: &ArchitectureSpec, arch2: &ArchitectureSpec) -> f64 {
    // Calculate structural distance between architectures
    let layer_count_diff = (arch1.layers.len() as i32 - arch2.layers.len() as i32).abs() as f64;
    let param_count_diff = (arch1.parameter_count() as i64 - arch2.parameter_count() as i64).abs() as f64;
    
    // Normalize and combine
    layer_count_diff + param_count_diff / 10000.0
}

/// NSGA-II implementation for multi-objective optimization
pub struct NSGA2State<T: Float> {
    /// Population
    pub population: Vec<ArchitectureCandidate>,
    
    /// Objectives
    pub objectives: Vec<String>,
    
    /// Non-dominated sorting
    pub fronts: Vec<Vec<usize>>,
    
    /// Crowding distances
    pub crowding_distances: HashMap<usize, f64>,
    
    /// Parameters
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + Default + std::fmt::Debug> NSGA2State<T> {
    pub fn new(objectives: Vec<String>) -> Self {
        Self {
            population: Vec::new(),
            objectives,
            fronts: Vec::new(),
            crowding_distances: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Perform non-dominated sorting
    pub fn non_dominated_sort(&mut self) {
        let n = self.population.len();
        let mut domination_count = vec![0; n];
        let mut dominated_solutions = vec![Vec::new(); n];
        self.fronts.clear();
        self.fronts.push(Vec::new());

        // Calculate domination relationships
        for i in 0..n {
            for j in 0..n {
                if i != j && self.dominates(i, j) {
                    dominated_solutions[i].push(j);
                } else if i != j && self.dominates(j, i) {
                    domination_count[i] += 1;
                }
            }

            if domination_count[i] == 0 {
                self.fronts[0].push(i);
            }
        }

        // Build subsequent fronts
        let mut front_idx = 0;
        while !self.fronts[front_idx].is_empty() {
            let mut next_front = Vec::new();
            
            for &individual in &self.fronts[front_idx] {
                for &dominated in &dominated_solutions[individual] {
                    domination_count[dominated] -= 1;
                    if domination_count[dominated] == 0 {
                        next_front.push(dominated);
                    }
                }
            }
            
            if !next_front.is_empty() {
                self.fronts.push(next_front);
            }
            front_idx += 1;
        }
    }

    /// Check if individual i dominates individual j
    fn dominates(&self, i: usize, j: usize) -> bool {
        let perf_i = &self.population[i].performance;
        let perf_j = &self.population[j].performance;

        let better_in_any = perf_i.optimization_performance > perf_j.optimization_performance
            || perf_i.convergence_speed > perf_j.convergence_speed
            || perf_i.generalization > perf_j.generalization;

        let worse_in_any = perf_i.optimization_performance < perf_j.optimization_performance
            || perf_i.convergence_speed < perf_j.convergence_speed
            || perf_i.generalization < perf_j.generalization;

        better_in_any && !worse_in_any
    }

    /// Calculate crowding distances
    pub fn calculate_crowding_distances(&mut self) {
        self.crowding_distances.clear();

        for front in &self.fronts {
            if front.len() <= 2 {
                for &individual in front {
                    self.crowding_distances.insert(individual, f64::INFINITY);
                }
                continue;
            }

            // Initialize distances
            for &individual in front {
                self.crowding_distances.insert(individual, 0.0);
            }

            // For each objective
            for obj_idx in 0..self.objectives.len() {
                let mut sorted_front = front.clone();
                sorted_front.sort_by(|&a, &b| {
                    self.get_objective_value(a, obj_idx)
                        .partial_cmp(&self.get_objective_value(b, obj_idx))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Set boundary points to infinite distance
                *self.crowding_distances.get_mut(&sorted_front[0]).unwrap() = f64::INFINITY;
                *self.crowding_distances.get_mut(&sorted_front[sorted_front.len() - 1]).unwrap() = f64::INFINITY;

                // Calculate distances for intermediate points
                let obj_range = self.get_objective_value(sorted_front[sorted_front.len() - 1], obj_idx)
                    - self.get_objective_value(sorted_front[0], obj_idx);

                if obj_range > 0.0 {
                    for i in 1..sorted_front.len() - 1 {
                        let distance_contrib = (self.get_objective_value(sorted_front[i + 1], obj_idx)
                            - self.get_objective_value(sorted_front[i - 1], obj_idx))
                            / obj_range;
                        
                        *self.crowding_distances.get_mut(&sorted_front[i]).unwrap() += distance_contrib;
                    }
                }
            }
        }
    }

    /// Get objective value for individual
    fn get_objective_value(&self, individual: usize, objective_idx: usize) -> f64 {
        let performance = &self.population[individual].performance;
        match objective_idx {
            0 => performance.optimization_performance,
            1 => performance.convergence_speed,
            2 => performance.generalization,
            3 => performance.robustness,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::architecture::*;

    #[test]
    fn test_evolutionary_state_creation() {
        let state: EvolutionarySearchState<f64> = EvolutionarySearchState::new(10);
        assert_eq!(state.parameters.population_size, 10);
        assert_eq!(state.generation, 0);
    }

    #[test]
    fn test_diversity_calculation() {
        let population = vec![
            ArchitectureCandidate::new(
                "arch1".to_string(),
                ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default()),
            ),
            ArchitectureCandidate::new(
                "arch2".to_string(),
                ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default()),
            ),
        ];

        let diversity = calculate_diversity_metrics(&population);
        assert!(diversity.structural_diversity >= 0.0);
        assert!(diversity.performance_diversity >= 0.0);
    }

    #[test]
    fn test_nsga2_state_creation() {
        let objectives = vec!["performance".to_string(), "efficiency".to_string()];
        let state: NSGA2State<f64> = NSGA2State::new(objectives);
        assert_eq!(state.objectives.len(), 2);
    }

    #[test]
    fn test_elite_selection() {
        let mut state: EvolutionarySearchState<f64> = EvolutionarySearchState::new(5);
        
        // Add some individuals
        for i in 0..5 {
            let mut candidate = ArchitectureCandidate::new(
                format!("arch{}", i),
                ArchitectureSpec::new(vec![], GlobalArchitectureConfig::default()),
            );
            candidate.performance.optimization_performance = i as f64 / 10.0;
            state.population.push(candidate);
        }

        let fitnesses: Vec<f64> = (0..5).map(|i| i as f64 / 10.0).collect();
        let elite = state.select_elite(&fitnesses);

        assert_eq!(elite.len(), state.parameters.elite_size);
        // Best individuals should be selected (highest indices have highest fitness)
        assert!(elite.contains(&4)); // Highest fitness
    }
}