//! Multi-objective optimization for Neural Architecture Search
//!
//! This module provides multi-objective optimization capabilities for NAS,
//! allowing optimization of multiple conflicting objectives simultaneously
//! such as accuracy, latency, FLOPs, memory usage, and energy consumption.

use crate::error::Result;
use crate::nas::{architecture_encoding::ArchitectureEncoding, EvaluationMetrics, SearchResult};
use std::collections::HashMap;
use std::sync::Arc;
/// Represents an objective to optimize
#[derive(Debug, Clone)]
pub struct Objective {
    /// Name of the objective
    pub name: String,
    /// Whether to minimize (true) or maximize (false) this objective
    pub minimize: bool,
    /// Weight for this objective in weighted sum approaches
    pub weight: f64,
    /// Target value for constraint handling
    pub target: Option<f64>,
    /// Tolerance for constraint satisfaction
    pub tolerance: Option<f64>,
}
impl Objective {
    /// Create a new objective
    pub fn new(name: &str, minimize: bool, weight: f64) -> Self {
        Self {
            _name: name.to_string(),
            minimize,
            weight,
            target: None,
            tolerance: None,
        }
    }
    /// Add constraint with target value and tolerance
    pub fn with_constraint(mut self, target: f64, tolerance: f64) -> Self {
        self.target = Some(target);
        self.tolerance = Some(tolerance);
        self
/// Multi-objective optimization configuration
pub struct MultiObjectiveConfig {
    /// List of objectives to optimize
    pub objectives: Vec<Objective>,
    /// Optimization algorithm to use
    pub algorithm: MultiObjectiveAlgorithm,
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Number of generations
    pub max_generations: usize,
    /// Pareto front size limit
    pub pareto_front_limit: usize,
    /// Hypervolume reference point
    pub reference_point: Option<Vec<f64>>,
impl Default for MultiObjectiveConfig {
    fn default() -> Self {
            objectives: vec![
                Objective::new("validation_accuracy", false, 0.4),
                Objective::new("model_flops", true, 0.3),
                Objective::new("model_params", true, 0.2),
                Objective::new("inference_latency", true, 0.1),
            ],
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            population_size: 50,
            max_generations: 100,
            pareto_front_limit: 20,
            reference_point: None,
/// Available multi-objective optimization algorithms
pub enum MultiObjectiveAlgorithm {
    /// Non-dominated Sorting Genetic Algorithm II
    NSGA2,
    /// Strength Pareto Evolutionary Algorithm 2
    SPEA2,
    /// Multi-Objective Evolutionary Algorithm based on Decomposition
    MOEAD,
    /// Hypervolume-based optimization
    HYPERE,
    /// Weighted sum approach
    WeightedSum,
    /// Constraint handling with objectives
    ConstraintHandling,
/// Solution in multi-objective space
pub struct MultiObjectiveSolution {
    /// Architecture encoding
    pub architecture: Arc<dyn ArchitectureEncoding>,
    /// Objective values
    pub objectives: Vec<f64>,
    /// Constraint violations
    pub constraint_violations: Vec<f64>,
    /// Rank in non-dominated sorting
    pub rank: usize,
    /// Crowding distance
    pub crowding_distance: f64,
    /// Dominance count
    pub dominance_count: usize,
    /// Solutions dominated by this solution
    pub dominated_solutions: Vec<usize>,
impl MultiObjectiveSolution {
    /// Create a new solution
    pub fn new(architecture: Arc<dyn ArchitectureEncoding>, objectives: Vec<f64>) -> Self {
            architecture,
            objectives,
            constraint_violations: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
            dominance_count: 0,
            dominated_solutions: Vec::new(),
    /// Check if this solution dominates another
    pub fn dominates(&self, other: &Self, config: &MultiObjectiveConfig) -> bool {
        let mut better_in_at_least_one = false;
        for (i, obj) in config.objectives.iter().enumerate() {
            let self_val = self.objectives[i];
            let other_val = other.objectives[i];
            if obj.minimize {
                if self_val > other_val {
                    return false; // Self is worse
                } else if self_val < other_val {
                    better_in_at_least_one = true;
                }
            } else {
                if self_val < other_val {
                } else if self_val > other_val {
            }
        better_in_at_least_one
/// Multi-objective optimizer
pub struct MultiObjectiveOptimizer {
    config: MultiObjectiveConfig,
    population: Vec<MultiObjectiveSolution>,
    pareto_front: Vec<MultiObjectiveSolution>,
    generation: usize,
    hypervolume_history: Vec<f64>,
impl MultiObjectiveOptimizer {
    /// Create a new multi-objective optimizer
    pub fn new(config: MultiObjectiveConfig) -> Self {
            config,
            population: Vec::new(),
            pareto_front: Vec::new(),
            generation: 0,
            hypervolume_history: Vec::new(),
    /// Initialize population from search results
    pub fn initialize_population(&mut self, results: &[SearchResult]) -> Result<()> {
        self.population.clear();
        for result in results.iter().take(self._config.population_size) {
            let objectives = self.extract_objectives(&result.metrics)?;
            let solution = MultiObjectiveSolution::new(result.architecture.clone(), objectives);
            self.population.push(solution);
        // Fill remaining population with random solutions if needed
        while self.population.len() < self._config.population_size {
            let random_arch = self.generate_random_architecture()?;
            let random_objectives = self.estimate_random_objectives();
            let solution = MultiObjectiveSolution::new(random_arch, random_objectives);
        Ok(())
    /// Run optimization for one generation
    pub fn evolve_generation(&mut self) -> Result<()> {
        match self._config.algorithm {
            MultiObjectiveAlgorithm::NSGA2 => self.nsga2_step()?,
            MultiObjectiveAlgorithm::SPEA2 => self.spea2_step()?,
            MultiObjectiveAlgorithm::MOEAD => self.moead_step()?,
            MultiObjectiveAlgorithm::HYPERE => self.hypere_step()?,
            MultiObjectiveAlgorithm::WeightedSum => self.weighted_sum_step()?,
            MultiObjectiveAlgorithm::ConstraintHandling => self.constraint_handling_step()?,
        self.generation += 1;
        // Update Pareto front
        self.update_pareto_front()?;
        // Compute hypervolume
        let hv = self.compute_hypervolume()?;
        self.hypervolume_history.push(hv);
    /// NSGA-II algorithm step
    fn nsga2_step(&mut self) -> Result<()> {
        // Create offspring through crossover and mutation
        let offspring = self.create_offspring()?;
        // Combine parent and offspring populations
        let mut combined_population = self.population.clone();
        combined_population.extend(offspring);
        // Non-dominated sorting
        self.non_dominated_sort(&mut combined_population)?;
        // Select next generation
        self.population = self.environmental_selection(combined_population)?;
    /// SPEA2 algorithm step
    fn spea2_step(&mut self) -> Result<()> {
        // Combine population and offspring
        // Calculate strength and raw fitness for all individuals
        self.calculate_spea2_fitness_for_population(&mut combined_population)?;
        // Environmental selection based on SPEA2 fitness
        self.population = self.spea2_environmental_selection(combined_population)?;
    /// MOEA/D algorithm step
    fn moead_step(&mut self) -> Result<()> {
        // Decompose problem into scalar subproblems
        let weight_vectors = self.generate_weight_vectors()?;
        // Update each subproblem
        for (i, weights) in weight_vectors.iter().enumerate() {
            if i < self.population.len() {
                let new_solution = self.update_subproblem(i, weights)?;
                // Update neighboring subproblems
                self.update_neighbors(i, &new_solution)?;
    /// Hypervolume-based optimization step
    fn hypere_step(&mut self) -> Result<()> {
        // Select parents based on hypervolume contribution
        let parents = self.hypervolume_selection()?;
        // Create offspring
        let offspring = self.create_offspring_from_parents(&parents)?;
        // Environmental selection based on hypervolume
        self.population = self.hypervolume_environmental_selection(offspring)?;
    /// Weighted sum approach step
    fn weighted_sum_step(&mut self) -> Result<()> {
        // Convert multi-objective to single objective using weights
        for solution in &mut self.population {
            let weighted_sum = solution
                .objectives
                .iter()
                .zip(self._config.objectives.iter())
                .map(|(obj_val, obj_config)| obj_val * obj_config.weight)
                .sum::<f64>();
            // Store as single objective
            solution.objectives = vec![weighted_sum];
        // Sort by weighted sum and select best
        self.population
            .sort_by(|a, b| a.objectives[0].partial_cmp(&b.objectives[0]).unwrap());
        // Create new population through mutation of best solutions
        self.population.extend(offspring);
        self.population.truncate(self._config.population_size);
    /// Constraint handling step
    fn constraint_handling_step(&mut self) -> Result<()> {
        // Evaluate constraints for each solution
            solution.constraint_violations = self.evaluate_constraints(solution)?;
        // Sort by constraint violation first, then by objectives
        self.population.sort_by(|a, b| {
            let a_violations: f64 = a.constraint_violations.iter().sum();
            let b_violations: f64 = b.constraint_violations.iter().sum();
            if a_violations != b_violations {
                a_violations.partial_cmp(&b_violations).unwrap()
                // Both feasible or equally infeasible, compare objectives
                self.compare_objectives(a, b)
        });
        // Create offspring and apply constraint handling
        // Select next generation with constraint preference
        self.population = self.constraint_environmental_selection(offspring)?;
    /// Non-dominated sorting algorithm
    fn non_dominated_sort(&self, population: &mut [MultiObjectiveSolution]) -> Result<()> {
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front = Vec::new();
        // Initialize domination relationships
        for (i, solution_i) in population.iter().enumerate() {
            for (j, solution_j) in population.iter().enumerate() {
                if i != j {
                    if solution_i.dominates(solution_j, &self.config) {
                        unsafe {
                            let solution_i_mut =
                                &mut *(solution_i as *const _ as *mut MultiObjectiveSolution);
                            solution_i_mut.dominated_solutions.push(j);
                        }
                    } else if solution_j.dominates(solution_i, &self.config) {
                            solution_i_mut.dominance_count += 1;
                    }
            if population[i].dominance_count == 0 {
                unsafe {
                    let solution_i_mut =
                        &mut *(population[i] as *const _ as *mut MultiObjectiveSolution);
                    solution_i_mut.rank = 0;
                current_front.push(i);
        fronts.push(current_front.clone());
        // Build subsequent fronts
        let mut front_index = 0;
        while !fronts[front_index].is_empty() {
            let mut next_front = Vec::new();
            for &i in &fronts[front_index] {
                for &j in &population[i].dominated_solutions {
                    unsafe {
                        let solution_j_mut =
                            &mut *(population[j] as *const _ as *mut MultiObjectiveSolution);
                        solution_j_mut.dominance_count -= 1;
                        if solution_j_mut.dominance_count == 0 {
                            solution_j_mut.rank = front_index + 1;
                            next_front.push(j);
            front_index += 1;
            fronts.push(next_front);
    /// Calculate crowding distance for diversity
    fn calculate_crowding_distance(
        &self,
        front: &[usize],
        population: &mut [MultiObjectiveSolution],
    ) -> Result<()> {
        if front.len() <= 2 {
            for &i in front {
                    let solution_mut =
                    solution_mut.crowding_distance = f64::INFINITY;
            return Ok(());
        // Initialize crowding distance
        for &i in front {
            unsafe {
                let solution_mut = &mut *(population[i] as *const _ as *mut MultiObjectiveSolution);
                solution_mut.crowding_distance = 0.0;
        // For each objective
        for obj_idx in 0..self.config.objectives.len() {
            // Sort by objective value
            let mut sorted_indices = front.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                population[a].objectives[obj_idx]
                    .partial_cmp(&population[b].objectives[obj_idx])
                    .unwrap()
            });
            // Set boundary points to infinity
                let first_mut = &mut *(population[sorted_indices[0]] as *const _
                    as *mut MultiObjectiveSolution);
                first_mut.crowding_distance = f64::INFINITY;
                let last_mut = &mut *(population[sorted_indices[sorted_indices.len() - 1]]
                    as *const _
                last_mut.crowding_distance = f64::INFINITY;
            // Calculate crowding distance for intermediate points
            let obj_min = population[sorted_indices[0]].objectives[obj_idx];
            let obj_max = population[sorted_indices[sorted_indices.len() - 1]].objectives[obj_idx];
            let obj_range = obj_max - obj_min;
            if obj_range > 0.0 {
                for i in 1..sorted_indices.len() - 1 {
                    let prev_obj = population[sorted_indices[i - 1]].objectives[obj_idx];
                    let next_obj = population[sorted_indices[i + 1]].objectives[obj_idx];
                        let solution_mut = &mut *(population[sorted_indices[i]] as *const _
                            as *mut MultiObjectiveSolution);
                        solution_mut.crowding_distance += (next_obj - prev_obj) / obj_range;
    /// Environmental selection for NSGA-II
    fn environmental_selection(
        mut population: Vec<MultiObjectiveSolution>,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut result = Vec::new();
        let mut current_front = 0;
        // Group solutions by rank
        let mut fronts: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, solution) in population.iter().enumerate() {
            fronts.entry(solution.rank).or_insert_with(Vec::new).push(i);
        // Add complete fronts
        while current_front < fronts.len() {
            if let Some(front) = fronts.get(&current_front) {
                if result.len() + front.len() <= self.config.population_size {
                    for &i in front {
                        result.push(population[i].clone());
                } else {
                    // Calculate crowding distance for the last front
                    self.calculate_crowding_distance(front, &mut population)?;
                    // Sort by crowding distance and add remaining solutions
                    let mut front_with_distance: Vec<_> = front
                        .iter()
                        .map(|&i| (i, population[i].crowding_distance))
                        .collect();
                    front_with_distance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let remaining = self.config.population_size - result.len();
                    for i in 0..remaining {
                        let idx = front_with_distance[i].0;
                        result.push(population[idx].clone());
                    break;
            current_front += 1;
        Ok(result)
    /// Create offspring through crossover and mutation
    fn create_offspring(&self) -> Result<Vec<MultiObjectiveSolution>> {
        let mut offspring = Vec::new();
        for _ in 0..self.config.population_size {
            // Tournament selection for parents
            let parent1 = self.tournament_selection()?;
            let parent2 = self.tournament_selection()?;
            // Crossover
            let child_arch = parent1
                .architecture
                .crossover(parent2.architecture.as_ref())?;
            // Mutation
            let mutated_arch = child_arch.mutate(0.1)?;
            // Evaluate objectives for offspring
            let objectives = self.estimate_objectives(&mutated_arch)?;
            let child = MultiObjectiveSolution::new(mutated_arch, objectives);
            offspring.push(child);
        Ok(offspring)
    /// Tournament selection for parent selection
    fn tournament_selection(&self) -> Result<&MultiObjectiveSolution> {
        use rand::prelude::*;
        let mut rng = rng();
        let tournament_size = 3;
        let mut best_idx = rng.gen_range(0..self.population.len());
        for _ in 1..tournament_size {
            let candidate_idx = rng.gen_range(0..self.population.len());
            // Compare based on dominance and crowding distance
            if self.is_better(&self.population[candidate_idx], &self.population[best_idx]) {
                best_idx = candidate_idx;
        Ok(&self.population[best_idx])
    /// Check if solution a is better than solution b
    fn is_better(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> bool {
        if a.rank < b.rank {
            true
        } else if a.rank > b.rank {
            false
        } else {
            // Same rank, compare crowding distance
            a.crowding_distance > b.crowding_distance
    /// Extract objectives from evaluation metrics
    fn extract_objectives(&self, metrics: &EvaluationMetrics) -> Result<Vec<f64>> {
        let mut objectives = Vec::new();
        for obj_config in &self.config.objectives {
            let value = metrics.get(&obj_config.name).copied().unwrap_or(0.0);
            objectives.push(value);
        Ok(objectives)
    /// Estimate objectives for an architecture
    fn estimate_objectives(
        architecture: &Arc<dyn ArchitectureEncoding>,
    ) -> Result<Vec<f64>> {
        // Simplified objective estimation
        // In practice, would use actual evaluation or prediction models
            let value = match obj_config.name.as_str() {
                "validation_accuracy" => 0.7 + 0.2 * rand::random::<f64>(),
                "model_flops" => 1e6 + 1e6 * rand::random::<f64>(),
                "model_params" => 1e5 + 1e5 * rand::random::<f64>(),
                "inference_latency" => 10.0 + 10.0 * rand::random::<f64>(, _ => 0.5,
            };
    /// Generate random architecture for initialization
    fn generate_random_architecture(&self) -> Result<Arc<dyn ArchitectureEncoding>> {
        let encoding = crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng)?;
        Ok(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>)
    /// Estimate random objectives
    fn estimate_random_objectives(&self) -> Vec<f64> {
        self.config
            .objectives
            .iter()
            .map(|obj| match obj.name.as_str() {
                "validation_accuracy" => 0.3 + 0.4 * rand::random::<f64>(),
                "model_flops" => 1e5 + 1e6 * rand::random::<f64>(),
                "model_params" => 1e4 + 1e5 * rand::random::<f64>(),
                "inference_latency" => 1.0 + 20.0 * rand::random::<f64>(, _ => rand::random::<f64>(),
            })
            .collect()
    /// Update Pareto front
    fn update_pareto_front(&mut self) -> Result<()> {
        // Find non-dominated solutions from current population
        let mut pareto_solutions = Vec::new();
        for solution in &self.population {
            let mut is_dominated = false;
            for other in &self.population {
                if other.dominates(solution, &self.config) {
                    is_dominated = true;
            if !is_dominated {
                pareto_solutions.push(solution.clone());
        // Limit Pareto front size
        if pareto_solutions.len() > self.config.pareto_front_limit {
            // Use crowding distance to select diverse solutions
            pareto_solutions.sort_by(|a, b| {
                b.crowding_distance
                    .partial_cmp(&a.crowding_distance)
            pareto_solutions.truncate(self.config.pareto_front_limit);
        self.pareto_front = pareto_solutions;
    /// Compute hypervolume indicator using the WFG algorithm for 2D and Monte Carlo for higher dimensions
    fn compute_hypervolume(&self) -> Result<f64> {
        if self.pareto_front.is_empty() {
            return Ok(0.0);
        let reference_point = self
            .config
            .reference_point
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.estimate_reference_point());
        if self.config.objectives.len() == 2 {
            self.compute_hypervolume_2d(&reference_point)
        } else if self.config.objectives.len() == 3 {
            self.compute_hypervolume_3d(&reference_point)
            self.compute_hypervolume_monte_carlo(&reference_point)
    /// Estimate reference point if not provided
    fn estimate_reference_point(&self) -> Vec<f64> {
        let mut reference = vec![0.0; self.config.objectives.len()];
        
        for (i, obj_config) in self.config.objectives.iter().enumerate() {
            if obj_config.minimize {
                // For minimization, use the maximum value found plus a margin
                let max_val = self.pareto_front
                    .iter()
                    .map(|s| s.objectives[i])
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(1.0);
                reference[i] = max_val * 1.1;
                // For maximization, use the minimum value found minus a margin
                let min_val = self.pareto_front
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);
                reference[i] = min_val * 0.9;
        reference
    /// Compute 2D hypervolume using the efficient sweep algorithm
    fn compute_hypervolume_2d(&self, referencepoint: &[f64]) -> Result<f64> {
        let mut points: Vec<(f64, f64)> = self.pareto_front
            .map(|s| {
                let x = if self.config.objectives[0].minimize {
                    reference_point[0] - s.objectives[0]
                    s.objectives[0] - reference_point[0]
                };
                let y = if self.config.objectives[1].minimize {
                    reference_point[1] - s.objectives[1]
                    s.objectives[1] - reference_point[1]
                (x.max(0.0), y.max(0.0))
            .collect();
        // Sort by x-coordinate (descending for maximization)
        points.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut volume = 0.0;
        let mut prev_y = 0.0;
        for (x, y) in points {
            if y > prev_y {
                volume += x * (y - prev_y);
                prev_y = y;
        Ok(volume)
    /// Compute 3D hypervolume using the WFG algorithm
    fn compute_hypervolume_3d(&self, referencepoint: &[f64]) -> Result<f64> {
        // Transform points relative to reference point
        let mut points: Vec<Vec<f64>> = self.pareto_front
                s.objectives
                    .enumerate()
                    .map(|(i, &obj_val)| {
                        if self.config.objectives[i].minimize {
                            (reference_point[i] - obj_val).max(0.0)
                        } else {
                            (obj_val - reference_point[i]).max(0.0)
                    })
                    .collect()
        // Sort points lexicographically
        points.sort_by(|a, b| {
            for (x, y) in a.iter().zip(b.iter()) {
                match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) => continue,
                    other => return other.unwrap_or(std::cmp::Ordering::Equal),
            std::cmp::Ordering::Equal
        // Use inclusion-exclusion principle for 3D
        let n = points.len();
        // Single points
        for point in &points {
            volume += point[0] * point[1] * point[2];
        // Subtract intersections of pairs
        for i in 0..n {
            for j in (i + 1)..n {
                let intersection = [
                    points[i][0].min(points[j][0]),
                    points[i][1].min(points[j][1]),
                    points[i][2].min(points[j][2]),
                ];
                volume -= intersection[0] * intersection[1] * intersection[2];
        // Add intersections of triples, etc. (simplified for demonstration)
        // This is a simplified implementation; full 3D hypervolume calculation is more complex
        Ok(volume.max(0.0))
    /// Compute hypervolume using Monte Carlo sampling for higher dimensions
    fn compute_hypervolume_monte_carlo(&self, referencepoint: &[f64]) -> Result<f64> {
        let num_samples = 100000;
        let mut dominated_count = 0;
        // Define the bounds for sampling
        let mut lower_bounds = vec![f64::INFINITY; self.config.objectives.len()];
        let mut upper_bounds = reference_point.to_vec();
        for solution in &self.pareto_front {
            for (i, &obj_val) in solution.objectives.iter().enumerate() {
                if self.config.objectives[i].minimize {
                    lower_bounds[i] = lower_bounds[i].min(obj_val);
                    upper_bounds[i] = upper_bounds[i].max(obj_val);
        // Monte Carlo sampling
        for _ in 0..num_samples {
            let mut sample_point = vec![0.0; self.config.objectives.len()];
            
            for i in 0..sample_point.len() {
                sample_point[i] = rng.gen_range(lower_bounds[i]..=upper_bounds[i]);
            // Check if sample point is dominated by any solution in Pareto front
            for solution in &self.pareto_front {
                let mut dominates = true;
                let mut better_in_one = false;
                for (i..(&sol_val, &sample_val)) in solution.objectives.iter().zip(sample_point.iter()).enumerate() {
                    if self.config.objectives[i].minimize {
                        if sol_val > sample_val {
                            dominates = false;
                            break;
                        } else if sol_val < sample_val {
                            better_in_one = true;
                    } else {
                        if sol_val < sample_val {
                        } else if sol_val > sample_val {
                if dominates && better_in_one {
            if is_dominated {
                dominated_count += 1;
        // Calculate volume
        let sampling_volume: f64 = upper_bounds
            .zip(lower_bounds.iter())
            .map(|(upper, lower)| upper - lower)
            .product();
        let hypervolume = sampling_volume * (dominated_count as f64 / num_samples as f64);
        Ok(hypervolume)
    /// Get current Pareto front
    pub fn get_pareto_front(&self) -> &[MultiObjectiveSolution] {
        &self.pareto_front
    /// Get hypervolume history
    pub fn get_hypervolume_history(&self) -> &[f64] {
        &self.hypervolume_history
    /// Get current generation
    pub fn get_generation(&self) -> usize {
        self.generation
    // Placeholder implementations for other algorithms
    /// Calculate SPEA2 fitness for all individuals in population
    fn calculate_spea2_fitness_for_population(
        let n = population.len();
        let mut strengths = vec![0; n];
        let mut raw_fitness = vec![0.0; n];
        let mut densities = vec![0.0; n];
        // Step 1: Calculate strength values
            let mut dominated_count = 0;
            for j in 0..n {
                if i != j && population[i].dominates(&population[j], &self.config) {
                    dominated_count += 1;
            strengths[i] = dominated_count;
        // Step 2: Calculate raw fitness
            let mut fitness = 0.0;
                if i != j && population[j].dominates(&population[i], &self.config) {
                    fitness += strengths[j] as f64;
            raw_fitness[i] = fitness;
        // Step 3: Calculate density (using k-th nearest neighbor)
        let k = (n as f64).sqrt() as usize;
            let mut distances = Vec::new();
                    let distance = self.euclidean_distance(&population[i], &population[j]);
                    distances.push(distance);
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let kth_distance = if k < distances.len() {
                distances[k - 1]
                distances.last().copied().unwrap_or(0.0)
            densities[i] = 1.0 / (kth_distance + 2.0);
        // Step 4: Calculate final fitness (raw fitness + density)
            population[i].crowding_distance = raw_fitness[i] + densities[i];
    /// SPEA2 environmental selection
    fn spea2_environmental_selection(
        // Sort by SPEA2 fitness (stored in crowding_distance field)
        population.sort_by(|a, b| a.crowding_distance.partial_cmp(&b.crowding_distance).unwrap());
        // Select the best individuals
        let mut selected = Vec::new();
        // First, add all non-dominated solutions (fitness < 1.0)
        for solution in &population {
            if solution.crowding_distance < 1.0 && selected.len() < self.config.population_size {
                selected.push(solution.clone());
        // If we need more solutions, add the best dominated ones
        if selected.len() < self.config.population_size {
            for solution in &population {
                if solution.crowding_distance >= 1.0 && selected.len() < self.config.population_size {
                    selected.push(solution.clone());
        // Truncate to population size
        selected.truncate(self.config.population_size);
        Ok(selected)
    /// Calculate Euclidean distance between two solutions in objective space
    fn euclidean_distance(&self, a: &MultiObjectiveSolution, b: &MultiObjectiveSolution) -> f64 {
        a.objectives
            .zip(b.objectives.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    /// Generate uniformly distributed weight vectors for MOEA/D
    fn generate_weight_vectors(&self) -> Result<Vec<Vec<f64>>> {
        let num_objectives = self.config.objectives.len();
        let num_weights = self.config.population_size;
        if num_objectives == 2 {
            // For 2 objectives, generate evenly spaced weights
            let mut weights = Vec::new();
            for i in 0..num_weights {
                let w1 = i as f64 / (num_weights - 1) as f64;
                let w2 = 1.0 - w1;
                weights.push(vec![w1, w2]);
            Ok(weights)
        } else if num_objectives == 3 {
            // For 3 objectives, use simplex lattice design
            let h = ((num_weights as f64).sqrt() as usize).max(1);
            for i in 0..=h {
                for j in 0..=(h - i) {
                    let k = h - i - j;
                    let w1 = i as f64 / h as f64;
                    let w2 = j as f64 / h as f64;
                    let w3 = k as f64 / h as f64;
                    weights.push(vec![w1, w2, w3]);
                    
                    if weights.len() >= num_weights {
                        break;
                if weights.len() >= num_weights {
            // Fill remaining with random weights if needed
            while weights.len() < num_weights {
                let mut weight = vec![0.0; num_objectives];
                let sum: f64 = (0..num_objectives).map(|_| rand::random::<f64>()).sum();
                for w in &mut weight {
                    *w = rand::random::<f64>() / sum;
                weights.push(weight);
            weights.truncate(num_weights);
            // For higher dimensions, use random weights
            for _ in 0..num_weights {
    /// Update a single subproblem in MOEA/D
    fn update_subproblem(&self, index: usize, weights: &[f64]) -> Result<MultiObjectiveSolution> {
        if index >= self.population.len() {
            return Err(crate::error::NeuralError::InvalidArgument(
                "Subproblem index out of bounds".to_string(),
            ));
        // Get current solution for this subproblem
        let current_solution = &self.population[index];
        // Create offspring through crossover with neighbors
        let parent1 = current_solution;
        let neighbor_idx = self.select_neighbor(index)?;
        let parent2 = &self.population[neighbor_idx];
        // Crossover
        let child_arch = parent1.architecture.crossover(parent2.architecture.as_ref())?;
        // Mutation
        let mutated_arch = child_arch.mutate(0.1)?;
        // Evaluate objectives
        let objectives = self.estimate_objectives(&mutated_arch)?;
        let mut child_solution = MultiObjectiveSolution::new(mutated_arch, objectives);
        // Calculate fitness using Tchebycheff approach
        let current_fitness = self.tchebycheff_fitness(&current_solution.objectives, weights);
        let child_fitness = self.tchebycheff_fitness(&child_solution.objectives, weights);
        // Return better solution
        if child_fitness < current_fitness {
            child_solution.crowding_distance = child_fitness;
            Ok(child_solution)
            let mut current_clone = current_solution.clone();
            current_clone.crowding_distance = current_fitness;
            Ok(current_clone)
    /// Select a neighbor for crossover in MOEA/D
    fn select_neighbor(&self, index: usize) -> Result<usize> {
        // For simplicity, select a random neighbor within a neighborhood
        let neighborhood_size = 10.min(self.population.len());
        let start = index.saturating_sub(neighborhood_size / 2);
        let end = (index + neighborhood_size / 2).min(self.population.len() - 1);
        let neighbor_idx = rng.gen_range(start..=end);
        if neighbor_idx == index && end > start {
            Ok(if neighbor_idx == start { end } else { start })
            Ok(neighbor_idx)
    /// Calculate Tchebycheff fitness for MOEA/D
    fn tchebycheff_fitness(&self..objectives: &[f64], weights: &[f64]) -> f64 {
        let mut max_weighted_diff = 0.0;
        for (i, (&obj_val, &weight)) in objectives.iter().zip(weights.iter()).enumerate() {
            let ideal_point = if self.config.objectives[i].minimize { 0.0 } else { 1.0 };
            let weighted_diff = weight * (obj_val - ideal_point).abs();
            max_weighted_diff = max_weighted_diff.max(weighted_diff);
        max_weighted_diff
    /// Update neighbors with new solution in MOEA/D
    fn update_neighbors(&mut self, index: usize, solution: &MultiObjectiveSolution) -> Result<()> {
        let end = (index + neighborhood_size / 2).min(self.population.len());
        for i in start..end {
            if i != index {
                // Compare solutions for this neighbor's subproblem
                let weight_vectors = self.generate_weight_vectors()?;
                if i < weight_vectors.len() {
                    let weights = &weight_vectors[i];
                    let current_fitness = self.tchebycheff_fitness(&self.population[i].objectives, weights);
                    let new_fitness = self.tchebycheff_fitness(&solution.objectives, weights);
                    if new_fitness < current_fitness {
                        // Update neighbor with better solution
                        self.population[i] = solution.clone();
    fn hypervolume_selection(&self) -> Result<Vec<&MultiObjectiveSolution>> {
        Ok(self.population.iter().take(10).collect())
    fn create_offspring_from_parents(
        parents: &[&MultiObjectiveSolution],
        self.create_offspring()
    fn hypervolume_environmental_selection(
        offspring: Vec<MultiObjectiveSolution>,
    fn evaluate_constraints(&self, solution: &MultiObjectiveSolution) -> Result<Vec<f64>> {
        let mut violations = Vec::new();
            if let (Some(target), Some(tolerance)) = (obj_config.target, obj_config.tolerance) {
                let violation = (solution.objectives[i] - target).abs() - tolerance;
                violations.push(violation.max(0.0));
        Ok(violations)
    fn compare_objectives(
        a: &MultiObjectiveSolution,
        b: &MultiObjectiveSolution,
    ) -> std::cmp::Ordering {
        // Simple comparison based on first objective
        a.objectives[0].partial_cmp(&b.objectives[0]).unwrap()
    fn constraint_environmental_selection(
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_multi_objective_config() {
        let config = MultiObjectiveConfig::default();
        assert_eq!(config.objectives.len(), 4);
        assert_eq!(config.population_size, 50);
    fn test_solution_dominance() {
        let arch1 = Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
            vec![],
        ));
        let arch2 = Arc::new(crate::nas::architecture_encoding::SequentialEncoding::new(
        let sol1 = MultiObjectiveSolution::new(arch1, vec![0.9, 1000.0, 500.0, 5.0]); // High acc, high cost
        let sol2 = MultiObjectiveSolution::new(arch2, vec![0.8, 500.0, 250.0, 2.5]); // Lower acc, lower cost
        // Neither should dominate the other (trade-off)
        assert!(!sol1.dominates(&sol2, &config));
        assert!(!sol2.dominates(&sol1, &config));
    fn test_optimizer_creation() {
        let optimizer = MultiObjectiveOptimizer::new(config);
        assert_eq!(optimizer.generation, 0);
        assert!(optimizer.pareto_front.is_empty());
