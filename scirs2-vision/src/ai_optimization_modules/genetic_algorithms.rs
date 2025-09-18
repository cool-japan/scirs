//! Genetic algorithms for pipeline evolution and optimization
//!
//! This module implements advanced genetic algorithms with multi-objective optimization,
//! adaptive mutation strategies, and elite archives for evolving computer vision pipelines.

use crate::error::Result;
use scirs2_core::{random::Random, rng};
use std::collections::{HashMap, VecDeque};

/// Advanced genetic algorithm for pipeline evolution with multi-objective optimization
pub struct GeneticPipelineOptimizer {
    /// Population of pipeline configurations
    pub population: Vec<PipelineGenome>,
    /// GA parameters
    ga_params: GAParameters,
    /// Fitness history
    fitness_history: VecDeque<GenerationStats>,
    /// Current generation
    current_generation: usize,
    /// Pareto front for multi-objective optimization
    pareto_front: Vec<PipelineGenome>,
    /// Adaptive mutation strategies
    adaptive_strategies: AdaptiveMutationStrategies,
    /// Elite archives for diversity preservation
    elite_archives: EliteArchives,
    /// Performance prediction models
    performance_predictors: PerformancePredictors,
}

/// Enhanced pipeline configuration genome with multi-objective fitness
#[derive(Debug, Clone)]
pub struct PipelineGenome {
    /// Pipeline parameters as genes
    pub genes: HashMap<String, f64>,
    /// Multi-objective fitness scores
    pub fitness_objectives: Vec<f64>,
    /// Aggregated fitness score
    pub fitness: f64,
    /// Age of the genome
    pub age: usize,
    /// Diversity contribution
    pub diversity_score: f64,
    /// Performance prediction confidence
    pub prediction_confidence: f64,
    /// Mutation strategy effectiveness
    pub mutation_effectiveness: f64,
}

/// Adaptive mutation strategies for enhanced evolution
#[derive(Debug, Clone)]
pub struct AdaptiveMutationStrategies {
    /// Gaussian mutation with adaptive sigma
    pub gaussian_sigma: f64,
    /// Polynomial mutation parameters
    pub polynomial_eta: f64,
    /// Differential evolution parameters
    pub de_f_factor: f64,
    /// Adaptive strategy weights
    pub strategy_weights: HashMap<MutationStrategy, f64>,
    /// Success tracking for each strategy
    pub strategy_success_rates: HashMap<MutationStrategy, f64>,
}

/// Available mutation strategies
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MutationStrategy {
    /// Gaussian mutation
    Gaussian,
    /// Polynomial mutation
    Polynomial,
    /// Differential evolution
    DifferentialEvolution,
    /// Cauchy mutation
    Cauchy,
    /// Lévy flight mutation
    LevyFlight,
    /// Self-adaptive mutation
    SelfAdaptive,
}

/// Elite archives for preserving diversity and high-quality solutions
#[derive(Debug, Clone)]
pub struct EliteArchives {
    /// High-performance solutions
    pub performance_archive: Vec<PipelineGenome>,
    /// Diverse solutions
    pub diversity_archive: Vec<PipelineGenome>,
    /// Novel solutions
    pub novelty_archive: Vec<PipelineGenome>,
    /// Archive capacity limits
    pub max_archive_size: usize,
}

/// Performance prediction models for guiding evolution
#[derive(Debug, Clone)]
pub struct PerformancePredictors {
    /// Neural network predictor for latency
    pub latency_predictor: NeuralNetworkPredictor,
    /// Neural network predictor for accuracy
    pub accuracy_predictor: NeuralNetworkPredictor,
    /// Neural network predictor for energy consumption
    pub energy_predictor: NeuralNetworkPredictor,
    /// Training data buffer
    pub training_buffer: VecDeque<(Vec<f64>, Vec<f64>)>,
    /// Model accuracy tracking
    pub prediction_accuracy: HashMap<String, f64>,
}

/// Simple neural network predictor
#[derive(Debug, Clone)]
pub struct NeuralNetworkPredictor {
    /// Input weights
    pub input_weights: Vec<Vec<f64>>,
    /// Hidden weights
    pub hidden_weights: Vec<Vec<f64>>,
    /// Output weights
    pub output_weights: Vec<f64>,
    /// Bias terms
    pub biases: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
}

impl NeuralNetworkPredictor {
    /// Create a new neural network predictor
    pub fn new(_input_size: usize, hidden_size: usize, outputsize: usize) -> Self {
        let mut rng = rng();

        // Initialize weights randomly
        let input_weights = (0..hidden_size)
            .map(|_| (0.._input_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();

        let hidden_weights = (0..outputsize)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-0.5..0.5)).collect())
            .collect();

        let output_weights = (0..outputsize).map(|_| rng.gen_range(-0.5..0.5)).collect();
        let biases = (0..hidden_size + outputsize)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            input_weights,
            hidden_weights,
            output_weights,
            biases,
            learning_rate: 0.01,
        }
    }

    /// Forward prediction
    pub fn predict(&self, input: &[f64]) -> Vec<f64> {
        // Forward pass through hidden layer
        let mut hidden_activations = Vec::new();
        for (i, weights) in self.input_weights.iter().enumerate() {
            let mut activation = self.biases[i];
            for (j, weight) in weights.iter().enumerate() {
                if j < input.len() {
                    activation += weight * input[j];
                }
            }
            hidden_activations.push(self.sigmoid(activation));
        }

        // Forward pass through output layer
        let mut output = Vec::new();
        for (i, weights) in self.hidden_weights.iter().enumerate() {
            let mut activation = self.biases[self.input_weights.len() + i];
            for (j, weight) in weights.iter().enumerate() {
                if j < hidden_activations.len() {
                    activation += weight * hidden_activations[j];
                }
            }
            output.push(activation); // Linear output for regression
        }

        output
    }

    /// Train one step using gradient descent
    pub fn train_step(&mut self, input: &[f64], target: f64, predicted: f64) {
        let error = target - predicted;

        // Simplified gradient descent - would use proper backpropagation in production
        let gradient_magnitude = error * self.learning_rate;

        // Update output weights
        for weight in &mut self.output_weights {
            *weight += gradient_magnitude * 0.1;
        }

        // Update hidden weights (simplified)
        for weights in &mut self.hidden_weights {
            for weight in weights {
                *weight += gradient_magnitude * 0.05;
            }
        }

        // Update input weights (simplified)
        for weights in &mut self.input_weights {
            for weight in weights {
                *weight += gradient_magnitude * 0.01;
            }
        }
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Gamma function approximation for Lévy flight sampling
#[allow(dead_code)]
fn gamma_function(x: f64) -> f64 {
    // Stirling's approximation for simplicity
    if x < 1.0 {
        return gamma_function(x + 1.0) / x;
    }

    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    sqrt_2pi / x.sqrt() * (x / std::f64::consts::E).powf(x)
}

/// Genetic algorithm parameters
#[derive(Debug, Clone)]
pub struct GAParameters {
    /// Population size
    pub populationsize: usize,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Crossover rate
    pub crossover_rate: f64,
    /// Elite selection ratio
    pub elite_ratio: f64,
    /// Maximum generations
    pub max_generations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for GAParameters {
    fn default() -> Self {
        Self {
            populationsize: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_ratio: 0.2,
            max_generations: 100,
            convergence_threshold: 0.001,
        }
    }
}

/// Statistics for a generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Generation number
    pub generation: usize,
    /// Best fitness in generation
    pub best_fitness: f64,
    /// Average fitness
    pub avg_fitness: f64,
    /// Worst fitness
    pub worst_fitness: f64,
    /// Diversity measure
    pub diversity: f64,
}

impl GeneticPipelineOptimizer {
    /// Create a new advanced genetic optimizer with multi-objective capabilities
    pub fn new(_parameterranges: HashMap<String, (f64, f64)>) -> Self {
        let ga_params = GAParameters::default();
        let population = Self::initialize_population(&_parameterranges, ga_params.populationsize);

        // Initialize adaptive mutation strategies
        let mut strategy_weights = HashMap::new();
        let mut strategy_success_rates = HashMap::new();
        for strategy in [
            MutationStrategy::Gaussian,
            MutationStrategy::Polynomial,
            MutationStrategy::DifferentialEvolution,
            MutationStrategy::Cauchy,
            MutationStrategy::LevyFlight,
            MutationStrategy::SelfAdaptive,
        ] {
            strategy_weights.insert(strategy.clone(), 1.0 / 6.0);
            strategy_success_rates.insert(strategy, 0.0);
        }

        let adaptive_strategies = AdaptiveMutationStrategies {
            gaussian_sigma: 0.1,
            polynomial_eta: 20.0,
            de_f_factor: 0.5,
            strategy_weights,
            strategy_success_rates,
        };

        let elite_archives = EliteArchives {
            performance_archive: Vec::new(),
            diversity_archive: Vec::new(),
            novelty_archive: Vec::new(),
            max_archive_size: 50,
        };

        let performance_predictors = PerformancePredictors {
            latency_predictor: NeuralNetworkPredictor::new(10, 8, 1),
            accuracy_predictor: NeuralNetworkPredictor::new(10, 8, 1),
            energy_predictor: NeuralNetworkPredictor::new(10, 8, 1),
            training_buffer: VecDeque::with_capacity(1000),
            prediction_accuracy: HashMap::new(),
        };

        Self {
            population,
            ga_params,
            fitness_history: VecDeque::with_capacity(1000),
            current_generation: 0,
            pareto_front: Vec::new(),
            adaptive_strategies,
            elite_archives,
            performance_predictors,
        }
    }

    /// Initialize random population with enhanced genomes
    fn initialize_population(
        parameter_ranges: &HashMap<String, (f64, f64)>,
        populationsize: usize,
    ) -> Vec<PipelineGenome> {
        let mut population = Vec::with_capacity(populationsize);
        let mut rng = rng();

        for _ in 0..populationsize {
            let mut genes = HashMap::new();

            for (param_name, &(min_val, max_val)) in parameter_ranges {
                let value = rng.gen_range(min_val..max_val + 1.0);
                genes.insert(param_name.clone(), value);
            }

            population.push(PipelineGenome {
                genes,
                fitness_objectives: vec![0.0; 5], // latency, accuracy, energy, memory, throughput
                fitness: 0.0,
                age: 0,
                diversity_score: 0.0,
                prediction_confidence: 0.0,
                mutation_effectiveness: 1.0,
            });
        }

        population
    }

    /// Perform advanced multi-objective evolution
    pub fn evolve_multi_objective(
        &mut self,
        fitness_evaluator: impl Fn(&PipelineGenome) -> Vec<f64>,
    ) -> Result<()> {
        // Evaluate all genomes
        for genome in &mut self.population {
            genome.fitness_objectives = fitness_evaluator(genome);
        }

        // Calculate fitness separately to avoid borrow conflicts
        let fitness_values: Vec<f64> = self
            .population
            .iter()
            .map(|genome| self.aggregate_objectives(&genome.fitness_objectives))
            .collect();

        for (genome, fitness) in self.population.iter_mut().zip(fitness_values) {
            genome.fitness = fitness;
        }

        // Update Pareto front
        self.update_pareto_front();

        // Perform selection, crossover, and adaptive mutation
        let new_population = self.adaptive_evolution()?;

        // Update archives
        self.update_elite_archives();

        // Train performance predictors
        self.train_performance_predictors()?;

        // Update adaptive strategies
        self.update_adaptive_strategies();

        self.population = new_population;
        self.current_generation += 1;

        Ok(())
    }

    /// Aggregate multiple objectives into a single fitness score
    fn aggregate_objectives(&self, objectives: &[f64]) -> f64 {
        // Weighted sum approach with adaptive weights
        let weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // latency, accuracy, energy, memory, throughput
        objectives
            .iter()
            .zip(weights.iter())
            .map(|(obj, weight)| obj * weight)
            .sum()
    }

    /// Update Pareto front with non-dominated solutions
    fn update_pareto_front(&mut self) {
        let mut new_front = Vec::new();

        for candidate in &self.population {
            let mut is_dominated = false;

            for existing in &self.pareto_front {
                if self.dominates(&existing.fitness_objectives, &candidate.fitness_objectives) {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                // Check which existing solutions are dominated by candidate
                let dominated_indices: Vec<usize> = self
                    .pareto_front
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, existing)| {
                        if self
                            .dominates(&candidate.fitness_objectives, &existing.fitness_objectives)
                        {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Remove dominated solutions (in reverse order to maintain indices)
                for &idx in dominated_indices.iter().rev() {
                    self.pareto_front.remove(idx);
                }

                new_front.push(candidate.clone());
            }
        }

        self.pareto_front.extend(new_front);

        // Limit Pareto front size
        if self.pareto_front.len() > 100 {
            self.pareto_front.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.pareto_front.truncate(100);
        }
    }

    /// Check if solution A dominates solution B (multi-objective)
    fn dominates(&self, a: &[f64], b: &[f64]) -> bool {
        let mut at_least_one_better = false;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            if a_val < b_val {
                return false; // A is worse in this objective
            }
            if a_val > b_val {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Perform adaptive evolution with multiple mutation strategies
    fn adaptive_evolution(&mut self) -> Result<Vec<PipelineGenome>> {
        let mut new_population = Vec::with_capacity(self.population.len());
        let mut rng = rng();

        // Keep elite solutions
        let elite_count = (self.population.len() as f64 * self.ga_params.elite_ratio) as usize;
        let mut sorted_pop = self.population.clone();
        sorted_pop.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for individual in sorted_pop.iter().take(elite_count) {
            new_population.push(individual.clone());
        }

        // Generate offspring using adaptive strategies
        while new_population.len() < self.population.len() {
            // Select parents using tournament selection
            let parent1 = self.tournament_selection(&mut rng);
            let parent2 = self.tournament_selection(&mut rng);

            // Crossover
            if rng.gen_range(0.0..1.0) < self.ga_params.crossover_rate {
                let (mut child1, mut child2) =
                    self.advanced_crossover(&parent1, &parent2, &mut rng);

                // Apply adaptive mutation
                self.adaptive_mutation(&mut child1, &mut rng)?;
                self.adaptive_mutation(&mut child2, &mut rng)?;

                new_population.push(child1);
                if new_population.len() < self.population.len() {
                    new_population.push(child2);
                }
            } else {
                new_population.push(parent1);
            }
        }

        Ok(new_population)
    }

    /// Tournament selection for parent selection
    fn tournament_selection(&self, rng: &mut Random) -> PipelineGenome {
        let tournament_size = 3;
        let mut best = &self.population[rng.gen_range(0..self.population.len())];

        for _ in 1..tournament_size {
            let candidate = &self.population[rng.gen_range(0..self.population.len())];
            if candidate.fitness > best.fitness {
                best = candidate;
            }
        }

        best.clone()
    }

    /// Advanced crossover combining multiple strategies
    fn advanced_crossover(
        &self,
        parent1: &PipelineGenome,
        parent2: &PipelineGenome,
        rng: &mut Random,
    ) -> (PipelineGenome, PipelineGenome) {
        let mut child1_genes = HashMap::new();
        let mut child2_genes = HashMap::new();

        for key in parent1.genes.keys() {
            let p1_val = parent1.genes[key];
            let p2_val = parent2.genes[key];

            // Simulated Binary Crossover (SBX)
            let eta = 20.0;
            let u = rng.gen_range(0.0..1.0);
            let beta = if u <= 0.5 {
                (2.0_f64 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0_f64 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };

            let c1 = 0.5 * ((1.0 + beta) * p1_val + (1.0 - beta) * p2_val);
            let c2 = 0.5 * ((1.0 - beta) * p1_val + (1.0 + beta) * p2_val);

            child1_genes.insert(key.clone(), c1);
            child2_genes.insert(key.clone(), c2);
        }

        let child1 = PipelineGenome {
            genes: child1_genes,
            fitness_objectives: vec![0.0; 5],
            fitness: 0.0,
            age: 0,
            diversity_score: 0.0,
            prediction_confidence: 0.0,
            mutation_effectiveness: (parent1.mutation_effectiveness
                + parent2.mutation_effectiveness)
                / 2.0,
        };

        let child2 = PipelineGenome {
            genes: child2_genes,
            fitness_objectives: vec![0.0; 5],
            fitness: 0.0,
            age: 0,
            diversity_score: 0.0,
            prediction_confidence: 0.0,
            mutation_effectiveness: (parent1.mutation_effectiveness
                + parent2.mutation_effectiveness)
                / 2.0,
        };

        (child1, child2)
    }

    /// Adaptive mutation using multiple strategies
    fn adaptive_mutation(&mut self, genome: &mut PipelineGenome, rng: &mut Random) -> Result<()> {
        // Select mutation strategy based on adaptive weights
        let strategy = self.select_mutation_strategy(rng);

        let mutation_strength = genome.mutation_effectiveness;

        for (_key, value) in genome.genes.iter_mut() {
            if rng.gen_range(0.0..1.0) < self.ga_params.mutation_rate * mutation_strength {
                match strategy {
                    MutationStrategy::Gaussian => {
                        let delta =
                            rng.gen_range(-1.0..1.0) * self.adaptive_strategies.gaussian_sigma;
                        *value += delta;
                    }
                    MutationStrategy::Polynomial => {
                        let eta = self.adaptive_strategies.polynomial_eta;
                        let u = rng.gen_range(0.0..1.0);
                        let delta = if u < 0.5 {
                            (2.0_f64 * u).powf(1.0 / (eta + 1.0)) - 1.0
                        } else {
                            1.0 - (2.0_f64 * (1.0 - u)).powf(1.0 / (eta + 1.0))
                        };
                        *value += delta * 0.1;
                    }
                    MutationStrategy::Cauchy => {
                        // Cauchy mutation with heavy tails
                        let cauchy_sample = (rng.gen_range(0.0..1.0) - 0.5) * std::f64::consts::PI;
                        let delta = cauchy_sample.tan() * 0.1;
                        *value += delta;
                    }
                    MutationStrategy::LevyFlight => {
                        // Lévy flight for exploration
                        let levy_sample = self.levy_flight_sample(rng);
                        *value += levy_sample * 0.1;
                    }
                    _ => {
                        // Default to Gaussian
                        let delta = rng.gen_range(-1.0..1.0) * 0.1;
                        *value += delta;
                    }
                }

                // Ensure bounds (simplified - would use actual parameter ranges)
                *value = value.clamp(0.0, 1.0);
            }
        }

        Ok(())
    }

    /// Select mutation strategy based on adaptive weights
    fn select_mutation_strategy(&self, rng: &mut Random) -> MutationStrategy {
        let mut cumulative_weight = 0.0;
        let random_value = rng.gen_range(0.0..1.0);

        for (strategy, weight) in &self.adaptive_strategies.strategy_weights {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                return strategy.clone();
            }
        }

        MutationStrategy::Gaussian // fallback
    }

    /// Generate Lévy flight sample
    fn levy_flight_sample(&self, rng: &mut Random) -> f64 {
        let beta = 1.5;
        let sigma_u = (gamma_function(1.0 + beta) * (beta * std::f64::consts::PI / 2.0).sin()
            / (gamma_function((1.0 + beta) / 2.0) * beta * (2.0_f64).powf((beta - 1.0) / 2.0)))
        .powf(1.0 / beta);

        let u = rng.gen_range(-1.0..1.0) * sigma_u;
        let v: f64 = rng.gen_range(-1.0..1.0);

        u / v.abs().powf(1.0 / beta)
    }

    /// Update elite archives with best and diverse solutions
    fn update_elite_archives(&mut self) {
        // Update performance archive
        let mut sorted_by_fitness = self.population.clone();
        sorted_by_fitness.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for genome in sorted_by_fitness.iter().take(10) {
            if self.elite_archives.performance_archive.len() < self.elite_archives.max_archive_size
            {
                self.elite_archives.performance_archive.push(genome.clone());
            }
        }

        // Update diversity archive (simplified diversity metric)
        for genome in &self.population {
            let diversity = self.calculate_diversity_score(genome);
            if diversity > 0.5
                && self.elite_archives.diversity_archive.len()
                    < self.elite_archives.max_archive_size
            {
                self.elite_archives.diversity_archive.push(genome.clone());
            }
        }
    }

    /// Calculate diversity score for a genome
    fn calculate_diversity_score(&self, genome: &PipelineGenome) -> f64 {
        let mut min_distance = f64::INFINITY;

        for other in &self.population {
            if std::ptr::eq(genome, other) {
                continue;
            }

            let distance = self.euclidean_distance(&genome.genes, &other.genes);
            min_distance = min_distance.min(distance);
        }

        min_distance
    }

    /// Calculate Euclidean distance between two genomes
    fn euclidean_distance(
        &self,
        genes1: &HashMap<String, f64>,
        genes2: &HashMap<String, f64>,
    ) -> f64 {
        let mut sum_squared_diff = 0.0;

        for (key, value1) in genes1 {
            if let Some(value2) = genes2.get(key) {
                sum_squared_diff += (value1 - value2).powi(2);
            }
        }

        sum_squared_diff.sqrt()
    }

    /// Train performance predictors using collected data
    fn train_performance_predictors(&mut self) -> Result<()> {
        if self.performance_predictors.training_buffer.len() < 10 {
            return Ok(()); // Need more data
        }

        // Train each predictor with a simple gradient descent step
        for (input, target) in self.performance_predictors.training_buffer.iter().take(50) {
            // Train latency predictor
            let predicted_latency = self.performance_predictors.latency_predictor.predict(input);
            self.performance_predictors.latency_predictor.train_step(
                input,
                target[0],
                predicted_latency[0],
            );

            // Train accuracy predictor
            let predicted_accuracy = self
                .performance_predictors
                .accuracy_predictor
                .predict(input);
            self.performance_predictors.accuracy_predictor.train_step(
                input,
                target[1],
                predicted_accuracy[0],
            );

            // Train energy predictor
            let predicted_energy = self.performance_predictors.energy_predictor.predict(input);
            self.performance_predictors.energy_predictor.train_step(
                input,
                target[2],
                predicted_energy[0],
            );
        }

        Ok(())
    }

    /// Update adaptive mutation strategy weights based on success rates
    fn update_adaptive_strategies(&mut self) {
        let total_success: f64 = self
            .adaptive_strategies
            .strategy_success_rates
            .values()
            .sum();

        if total_success > 0.0 {
            for (strategy, weight) in self.adaptive_strategies.strategy_weights.iter_mut() {
                let success_rate = self.adaptive_strategies.strategy_success_rates[strategy];
                *weight = success_rate / total_success;
            }
        }
    }

    /// Evaluate fitness of entire population
    pub fn evaluate_population(&mut self, fitnessfn: impl Fn(&PipelineGenome) -> f64) {
        for genome in &mut self.population {
            genome.fitness = fitnessfn(genome);
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Record generation statistics
        let best_fitness = self.population[0].fitness;
        let worst_fitness = self
            .population
            .last()
            .expect("Population should not be empty")
            .fitness;
        let avg_fitness =
            self.population.iter().map(|g| g.fitness).sum::<f64>() / self.population.len() as f64;

        let diversity = self.calculate_diversity();

        self.fitness_history.push_back(GenerationStats {
            generation: self.current_generation,
            best_fitness,
            avg_fitness,
            worst_fitness,
            diversity,
        });

        if self.fitness_history.len() > 1000 {
            self.fitness_history.pop_front();
        }
    }

    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        let mut total_distance = 0.0;
        let mut comparisons = 0;

        for i in 0..self.population.len() {
            for j in (i + 1)..self.population.len() {
                let distance = self.genome_distance(&self.population[i], &self.population[j]);
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

    /// Calculate distance between two genomes
    fn genome_distance(&self, genome1: &PipelineGenome, genome2: &PipelineGenome) -> f64 {
        let mut distance = 0.0;
        let mut count = 0;

        for (param_name, &value1) in &genome1.genes {
            if let Some(&value2) = genome2.genes.get(param_name) {
                distance += (value1 - value2).abs();
                count += 1;
            }
        }

        if count > 0 {
            distance / count as f64
        } else {
            0.0
        }
    }

    /// Evolve population for one generation
    pub fn evolve_generation(&mut self) -> bool {
        let elite_count = (self.population.len() as f64 * self.ga_params.elite_ratio) as usize;
        let mut new_population = Vec::with_capacity(self.population.len());
        let mut rng = rng();

        // Keep elite individuals
        for i in 0..elite_count {
            let mut elite = self.population[i].clone();
            elite.age += 1;
            new_population.push(elite);
        }

        // Generate offspring
        while new_population.len() < self.population.len() {
            // Tournament selection
            let parent1 = self.tournament_selection(&mut rng);
            let parent2 = self.tournament_selection(&mut rng);

            // Crossover
            let mut offspring = if rng.random_f64() < self.ga_params.crossover_rate {
                self.crossover(&parent1, &parent2)
            } else {
                parent1.clone()
            };

            // Mutation
            if rng.random_f64() < self.ga_params.mutation_rate {
                self.mutate(&mut offspring);
            }

            offspring.age = 0;
            offspring.fitness = 0.0;
            new_population.push(offspring);
        }

        self.population = new_population;
        self.current_generation += 1;

        // Check convergence
        self.check_convergence()
    }

    /// Single-point crossover
    fn crossover(&self, parent1: &PipelineGenome, parent2: &PipelineGenome) -> PipelineGenome {
        let mut offspring_genes = HashMap::new();
        let mut rng = rng();

        for (param_name, &value1) in &parent1.genes {
            if let Some(&value2) = parent2.genes.get(param_name) {
                let offspring_value = if rng.random_f64() < 0.5 {
                    value1
                } else {
                    value2
                };
                offspring_genes.insert(param_name.clone(), offspring_value);
            } else {
                offspring_genes.insert(param_name.clone(), value1);
            }
        }

        PipelineGenome {
            genes: offspring_genes,
            fitness_objectives: vec![0.0; 5],
            fitness: 0.0,
            age: 0,
            diversity_score: 0.0,
            prediction_confidence: 0.0,
            mutation_effectiveness: 1.0,
        }
    }

    /// Gaussian mutation
    fn mutate(&self, genome: &mut PipelineGenome) {
        let mut rng = rng();
        let mutation_strength = 0.1;

        for value in genome.genes.values_mut() {
            let mutation = rng.gen_range(-mutation_strength..mutation_strength + f64::EPSILON);
            *value += mutation;
            *value = value.clamp(0.0, 1.0); // Keep in valid range
        }
    }

    /// Check if algorithm has converged
    fn check_convergence(&self) -> bool {
        if self.fitness_history.len() < 10 {
            return false;
        }

        let recent_best: Vec<f64> = self
            .fitness_history
            .iter()
            .rev()
            .take(10)
            .map(|stats| stats.best_fitness)
            .collect();

        let variance = {
            let mean = recent_best.iter().sum::<f64>() / recent_best.len() as f64;
            recent_best.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / recent_best.len() as f64
        };

        variance < self.ga_params.convergence_threshold
    }

    /// Get best genome
    pub fn get_best_genome(&self) -> &PipelineGenome {
        &self.population[0]
    }

    /// Get generation statistics
    pub fn get_generation_stats(&self) -> Vec<GenerationStats> {
        self.fitness_history.iter().cloned().collect()
    }
}
