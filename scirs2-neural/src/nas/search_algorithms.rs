//! Search algorithms for Neural Architecture Search

use crate::error::Result;
use crate::nas::architecture__encoding::ArchitectureEncoding;
use crate::nas::SearchResult;
use ndarray::prelude::*;
use ndarray::{s, Array1, Array2};
use std::sync::Arc;
/// Trait for search algorithms
pub trait SearchAlgorithm: Send + Sync {
    /// Propose architectures to evaluate
    fn propose_architectures(
        &self,
        history: &[SearchResult],
        n_proposals: usize,
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>>;
    /// Update the algorithm with new results
    fn update(&mut self, results: &[SearchResult]) -> Result<()>;
    /// Get algorithm name
    fn name(&self) -> &str;
}
/// Random search algorithm
pub struct RandomSearch {
    seed: Option<u64>,
impl RandomSearch {
    /// Create a new random search algorithm
    pub fn new() -> Self {
        Self { seed: None }
    }
    /// Create with a specific seed
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(_seed) }
impl SearchAlgorithm for RandomSearch {
        _history: &[SearchResult],
    ) -> Result<Vec<Arc<dyn ArchitectureEncoding>>> {
        use rand::prelude::*;
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut rng()).unwrap()
        };
        let mut proposals = Vec::with_capacity(n_proposals);
        for _ in 0..n_proposals {
            // Create random architecture encoding
            let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng)?;
            proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
        }
        Ok(proposals)
    fn update(&mut selfresults: &[SearchResult]) -> Result<()> {
        // Random search doesn't learn from history
        Ok(())
    fn name(&self) -> &str {
        "RandomSearch"
/// Evolutionary search algorithm
pub struct EvolutionarySearch {
    population_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
    tournament_size: usize,
    elite_size: usize,
    population: Vec<Arc<dyn ArchitectureEncoding>>,
    fitness_scores: Vec<f64>,
impl EvolutionarySearch {
    /// Create a new evolutionary search algorithm
    pub fn new(_populationsize: usize) -> Self {
        Self {
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.9,
            tournament_size: 3,
            elite_size: _population_size / 10,
            population: Vec::new(),
            fitness_scores: Vec::new(),
    /// Set mutation rate
    pub fn with_mutation_rate(mut self, rate: f32) -> Self {
        self.mutation_rate = rate;
        self
    /// Set crossover rate
    pub fn with_crossover_rate(mut self, rate: f32) -> Self {
        self.crossover_rate = rate;
    /// Tournament selection
    fn tournament_select(&self, rng: &mut impl rand::Rng) -> usize {
        let mut best_idx = rng.gen_range(0..self.population.len());
        let mut best_fitness = self.fitness_scores[best_idx];
        for _ in 1..self.tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            if self.fitness_scores[idx] > best_fitness {
                best_idx = idx;
                best_fitness = self.fitness_scores[idx];
            }
        best_idx
impl SearchAlgorithm for EvolutionarySearch {
        let mut rng = rng();
        // Initialize population if empty
        if self.population.is_empty() {
            let mut proposals = Vec::with_capacity(n_proposals);
            for _ in 0..n_proposals {
                let encoding = crate::nas::architecture_encoding::GraphEncoding::random(&mut rng)?;
                proposals.push(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>);
            return Ok(proposals);
        // Elite selection
        let mut elite_indices: Vec<usize> = (0..self.population.len()).collect();
        elite_indices.sort_by(|&a, &b| {
            self.fitness_scores[b]
                .partial_cmp(&self.fitness_scores[a])
                .unwrap()
        });
        for &idx in elite_indices.iter().take(self.elite_size.min(n_proposals)) {
            proposals.push(self.population[idx].clone());
        // Generate offspring
        while proposals.len() < n_proposals {
            if rng.random::<f32>() < self.crossover_rate && self.population.len() >= 2 {
                // Crossover
                let parent1_idx = self.tournament_select(&mut rng);
                let parent2_idx = self.tournament_select(&mut rng);
                if parent1_idx != parent2_idx {
                    let offspring = self.population[parent1_idx]
                        .crossover(self.population[parent2_idx].as_ref())?;
                    proposals.push(Arc::from(offspring));
                } else {
                    // Fallback to mutation
                    let parent_idx = self.tournament_select(&mut rng);
                    let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                }
            } else {
                // Mutation
                let parent_idx = self.tournament_select(&mut rng);
                let offspring = self.population[parent_idx].mutate(self.mutation_rate)?;
                proposals.push(offspring);
    fn update(&mut self, results: &[SearchResult]) -> Result<()> {
        // Update population with new results
        for result in results {
            self.population.push(result.architecture.clone());
            let fitness = result.metrics.values().sum::<f64>() / result.metrics.len() as f64;
            self.fitness_scores.push(fitness);
        // Trim population to size
        if self.population.len() > self.population_size {
            let mut indices: Vec<usize> = (0..self.population.len()).collect();
            indices.sort_by(|&a, &b| {
                self.fitness_scores[b]
                    .partial_cmp(&self.fitness_scores[a])
                    .unwrap()
            });
            let new_population: Vec<_> = indices
                .iter()
                .take(self.population_size)
                .map(|&idx| self.population[idx].clone())
                .collect();
            let new_scores: Vec<_> = indices
                .map(|&idx| self.fitness_scores[idx])
            self.population = new_population;
            self.fitness_scores = new_scores;
        "EvolutionarySearch"
/// Reinforcement learning based search with REINFORCE controller
pub struct ReinforcementSearch {
    controller_hidden_size: usize,
    learning_rate: f32,
    entropy_weight: f32,
    baseline_decay: f32,
    baseline: Option<f64>,
    /// RNN controller for sequence generation
    controller_network: Option<ControllerNetwork>,
    /// Architecture generation history
    generation_history: Vec<Vec<f32>>,
/// Simple RNN controller for architecture generation
struct ControllerNetwork {
    hidden_size: usize,
    embedding_dim: usize,
    /// Weights for embedding layer
    embedding_weights: Array2<f32>,
    /// RNN cell weights
    rnn_weights: Array2<f32>,
    /// Output layer weights  
    output_weights: Array2<f32>,
    /// Hidden state
    hidden_state: Array1<f32>,
impl ReinforcementSearch {
    /// Create a new reinforcement learning search
            controller_hidden_size: 100,
            learning_rate: 3.5e-4,
            entropy_weight: 0.01,
            baseline_decay: 0.99,
            baseline: None,
            controller_network: None,
            generation_history: Vec::new(),
    /// Initialize the controller network
    fn initialize_controller(&mut self) -> Result<()> {
        let embedding_dim = 32;
        let vocab_size = 50; // Number of possible architecture choices
        // Initialize embedding weights
        let embedding_weights = Array2::random(
            (vocab_size, embedding_dim),
            rand_distr::Normal::new(0.0, 0.1).unwrap(),
        );
        // Initialize RNN weights
        let rnn_weights = Array2::random(
            (
                embedding_dim + self.controller_hidden_size,
                self.controller_hidden_size,
            ),
        // Initialize output weights
        let output_weights = Array2::random(
            (self.controller_hidden_size, vocab_size),
        let hidden_state = Array1::zeros(self.controller_hidden_size);
        self.controller_network = Some(ControllerNetwork {
            hidden_size: self.controller_hidden_size,
            embedding_dim,
            embedding_weights,
            rnn_weights,
            output_weights,
            hidden_state,
    /// Generate architecture sequence using REINFORCE
    fn generate_architecture_sequence(&mut self) -> Result<Vec<usize>> {
        if self.controller_network.is_none() {
            self.initialize_controller()?;
        let network = self.controller_network.as_mut().unwrap();
        let mut sequence = Vec::new();
        let mut log_probs = Vec::new();
        // Reset hidden state
        network.hidden_state.fill(0.0);
        // Generate sequence of architecture decisions
        for step in 0..20 {
            // Max sequence length
            let input_token = if step == 0 {
                0
                *sequence.last().unwrap()
            };
            // Forward pass through controller
            let (next_token, log_prob) = self.controller_forward_step(input_token)?;
            sequence.push(next_token);
            log_probs.push(log_prob);
            // Stop token
            if next_token == 0 {
                break;
        // Store for training
        self.generation_history.push(log_probs);
        Ok(sequence)
    /// Single forward step through controller
    fn controller_forward_step(&mut self, inputtoken: usize) -> Result<(usize, f32)> {
        // Embedding lookup
        let embedding = network
            .embedding_weights
            .row(input_token.min(network.embeddingweights.nrows() - 1));
        // Concatenate embedding with hidden state
        let mut rnn_input = Array1::zeros(network.embedding_dim + network.hidden_size);
        rnn_input
            .slice_mut(s![..network.embedding_dim])
            .assign(&embedding);
            .slice_mut(s![network.embedding_dim..])
            .assign(&network.hidden_state);
        // RNN forward pass (simplified)
        let rnn_output = rnn_input.dot(&network.rnn_weights);
        // Apply tanh activation
        network.hidden_state = rnn_output.mapv(|x| x.tanh());
        // Output layer
        let logits = network.hidden_state.dot(&network.output_weights);
        // Softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Array1<f32> = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        let probs = exp_logits.mapv(|x| x / sum_exp);
        // Sample from distribution
        let random_val: f32 = rng.random();
        let mut cumsum = 0.0;
        let mut selected_token = 0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                selected_token = i;
        let log_prob = probs[selected_token].ln();
        Ok((selected_token, log_prob))
    /// Update controller using REINFORCE
    fn update_controller(&mut self, rewards: &[f64]) -> Result<()> {
        if self.controller_network.is_none() || self.generation_history.is_empty() {
            return Ok(());
        // Compute advantage using baseline
        let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let baseline = self.baseline.unwrap_or(mean_reward);
        let advantages: Vec<f64> = rewards.iter().map(|r| r - baseline).collect();
        // Compute policy gradients
        for (history, advantage) in self.generation_history.iter().zip(advantages.iter()) {
            // Simplified gradient computation
            // In practice, would use proper backpropagation
            let gradient_scale = (*advantage as f32) * self.learning_rate;
            // Update controller weights (simplified)
            if let Some(ref mut network) = self.controller_network {
                // Apply gradient updates to weights
                network
                    .output_weights
                    .mapv_inplace(|w| w + gradient_scale * 0.001);
                    .rnn_weights
        // Clear history
        self.generation_history.clear();
impl SearchAlgorithm for ReinforcementSearch {
        // Cast to mutable for controller operations
        let self_mut = unsafe { &mut *(self as *const Self as *mut Self) };
            // Generate architecture using controller if available
            let encoding = if self.controller_network.is_some() {
                let sequence = self_mut.generate_architecture_sequence()?;
                // Convert sequence to architecture encoding
                self_mut.sequence_to_encoding(&sequence)?
                // Fallback to random generation
                use rand::prelude::*;
                let mut rng = rng();
                Arc::new(encoding) as Arc<dyn ArchitectureEncoding>
            proposals.push(encoding);
        // Extract rewards from results
        let rewards: Vec<f64> = results
            .iter()
            .map(|r| r.metrics.values().sum::<f64>() / r.metrics.len() as f64)
            .collect();
        if rewards.is_empty() {
        let mean_reward = rewards.iter().copied().sum::<f64>() / rewards.len() as f64;
        // Update baseline with exponential moving average
        self.baseline = Some(match self.baseline {
            Some(b) => {
                self.baseline_decay as f64 * b + (1.0 - self.baseline_decay as f64) * mean_reward
            None => mean_reward,
        // Update controller network using REINFORCE
        self.update_controller(&rewards)?;
        "ReinforcementSearch"
    /// Convert sequence to architecture encoding
    fn sequence_to_encoding(&self, sequence: &[usize]) -> Result<Arc<dyn ArchitectureEncoding>> {
        // Convert sequence tokens to layer types
        let mut layers = Vec::new();
        for &token in sequence {
            let layer_type = match token % 7 {
                0 => continue, // Skip/end token
                1 => crate::nas::search_space::LayerType::Dense(64 + (token % 4) * 64),
                2 => crate::nas::search_space::LayerType::Conv2D {
                    filters: 32 + (token % 4) * 32,
                    kernel_size: (3, 3),
                    stride: (1, 1),
                },
                3 => crate::nas::search_space::LayerType::Dropout(0.1 + (token % 4) as f32 * 0.1),
                4 => crate::nas::search_space::LayerType::BatchNorm,
                5 => crate::nas::search_space::LayerType::Activation("relu".to_string(), _ => crate::nas::search_space::LayerType::MaxPool2D {
                    pool_size: (2, 2),
                    stride: (2, 2),
            layers.push(layer_type);
            if layers.len() >= 15 {
                // Max layers
        // Create sequential encoding
        let encoding = crate::nas::architecture_encoding::SequentialEncoding::new(layers);
        Ok(Arc::new(encoding) as Arc<dyn ArchitectureEncoding>)
/// Differentiable architecture search (DARTS)
pub struct DifferentiableSearch {
    temperature: f64,
    arch_learning_rate: f32,
    weight_learning_rate: f32,
    arch_weight_decay: f32,
    /// Architecture parameters (alpha)
    alpha_normal: Option<Array2<f32>>,
    /// Architecture parameters for reduction cells
    alpha_reduce: Option<Array2<f32>>,
    /// Mixed operations for continuous relaxation
    mixed_ops: Vec<String>,
    /// Number of intermediate nodes
    num_intermediate_nodes: usize,
    /// Current epoch for progressive shrinking
    current_epoch: usize,
impl DifferentiableSearch {
    /// Create a new differentiable search
            temperature: 1.0,
            arch_learning_rate: 3e-4,
            weight_learning_rate: 0.025,
            arch_weight_decay: 1e-3,
            alpha_normal: None,
            alpha_reduce: None,
            mixed_ops: vec![
                "none".to_string(),
                "max_pool_3x3".to_string(),
                "avg_pool_3x3".to_string(),
                "skip_connect".to_string(),
                "sep_conv_3x3".to_string(),
                "sep_conv_5x5".to_string(),
                "dil_conv_3x3".to_string(),
                "dil_conv_5x5".to_string(),
            ],
            num_intermediate_nodes: 4,
            current_epoch: 0,
    /// Initialize architecture parameters
    fn initialize_alphas(&mut self) -> Result<()> {
        let num_ops = self.mixed_ops.len();
        let num_edges = self.num_intermediate_nodes * (self.num_intermediate_nodes + 1) / 2;
        // Initialize with small random values
        let alpha_normal = Array2::random(
            (num_edges, num_ops),
            rand_distr::Normal::new(0.0, 0.001).unwrap(),
        let alpha_reduce = Array2::random(
        self.alpha_normal = Some(alpha_normal);
        self.alpha_reduce = Some(alpha_reduce);
    /// Apply Gumbel softmax for continuous relaxation
    fn gumbel_softmax(&self, logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
        // Add Gumbel noise
        let gumbel_noise: Array1<f32> = Array1::from_shape_fn(logits.len(), |_| {
            let u: f32 = rng.random();
            -((-u.ln()).ln())
        let noisy_logits = logits + &gumbel_noise;
        // Apply softmax with temperature
        let scaled_logits = noisy_logits.mapv(|x| x / temperature);
        let max_logit = scaled_logits
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
        exp_logits.mapv(|x| x / sum_exp)
    /// Sample architecture from continuous distribution
    fn sample_architecture(
        &mut self,
    ) -> Result<crate::nas::architecture_encoding::SequentialEncoding> {
        if self.alpha_normal.is_none() {
            self.initialize_alphas()?;
        let alpha = self.alpha_normal.as_ref().unwrap();
        // Sample operations for each edge
        for edge_idx in 0..alpha.nrows() {
            let logits = alpha.row(edge_idx).to_owned();
            let probs = self.gumbel_softmax(&logits, self.temperature as f32);
            // Find the operation with highest probability
            let mut max_prob = 0.0;
            let mut selected_op = 0;
            for (i, &prob) in probs.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    selected_op = i;
            // Convert operation to layer type
            if selected_op > 0 {
                // Skip "none" operations
                let layer_type = self.operation_to_layer_type(selectedop)?;
                layers.push(layer_type);
        // Ensure we have at least a few layers
        if layers.len() < 3 {
            layers.push(crate::nas::search_space::LayerType::Dense(128));
            layers.push(crate::nas::search_space::LayerType::Activation(
                "relu".to_string(),
            ));
            layers.push(crate::nas::search_space::LayerType::Dense(64));
        Ok(crate::nas::architecture_encoding::SequentialEncoding::new(
            layers,
        ))
    /// Convert operation index to layer type
    fn operation_to_layer_type(
        op_idx: usize,
    ) -> Result<crate::nas::search_space::LayerType> {
        let layer_type = match self.mixed_ops.get(op_idx) {
            Some(op) => match op.as_str() {
                "none" => {
                    return Err(crate::error::NeuralError::InvalidArgument(
                        "None operation".to_string(),
                    ))
                "max_pool_3x3" => crate::nas::search_space::LayerType::MaxPool2D {
                    pool_size: (3, 3),
                "avg_pool_3x3" => crate::nas::search_space::LayerType::AvgPool2D {
                "skip_connect" => crate::nas::search_space::LayerType::Residual,
                "sep_conv_3x3" => crate::nas::search_space::LayerType::Conv2D {
                    filters: 64,
                "sep_conv_5x5" => crate::nas::search_space::LayerType::Conv2D {
                    kernel_size: (5, 5),
                "dil_conv_3x3" => crate::nas::search_space::LayerType::Conv2D {
                "dil_conv_5x5" => crate::nas::search_space::LayerType::Conv2D {
                _ => crate::nas::search_space::LayerType::Dense(64),
            },
            None => crate::nas::search_space::LayerType::Dense(64),
        Ok(layer_type)
    /// Update architecture parameters using gradient descent
    fn update_alphas(&mut self, validationloss: f64) -> Result<()> {
        if let Some(ref mut alpha) = self.alpha_normal {
            // Simplified gradient update
            // In practice, would compute actual gradients
            let gradient_scale = self.arch_learning_rate * validation_loss as f32;
            // Add regularization (weight decay)
            alpha.mapv_inplace(|x| x * (1.0 - self.arch_weight_decay) - gradient_scale * 0.001);
        if let Some(ref mut alpha) = self.alpha_reduce {
    /// Progressive shrinking of temperature
    fn update_temperature(&mut self) {
        self.current_epoch += 1;
        // Exponential decay of temperature
        self.temperature = (self.temperature * 0.98).max(0.1);
impl SearchAlgorithm for DifferentiableSearch {
        // Cast to mutable for alpha operations
            let encoding = self_mut.sample_architecture()?;
        if results.is_empty() {
        // Compute average validation loss
        let avg_loss = results
            .filter_map(|r| r.metrics.get("validation_loss"))
            .sum::<f64>()
            / results.len() as f64;
        // Update architecture parameters
        self.update_alphas(avg_loss)?;
        // Update temperature for next iteration
        self.update_temperature();
        "DifferentiableSearch"
/// Bayesian optimization for architecture search
pub struct BayesianOptimization {
    surrogate_type: String,
    acquisition_function: String,
    n_initial_points: usize,
    xi: f64, // Exploration parameter
impl BayesianOptimization {
    /// Create a new Bayesian optimization search
            surrogate_type: "gaussian_process".to_string(),
            acquisition_function: "expected_improvement".to_string(),
            n_initial_points: 10,
            xi: 0.01,
    /// Set acquisition function
    pub fn with_acquisition(mut self, acquisition: &str) -> Self {
        self.acquisition_function = acquisition.to_string();
impl SearchAlgorithm for BayesianOptimization {
        // If not enough initial points, do random search
        if history.len() < self.n_initial_points {
                let encoding =
                    crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng)?;
        // In practice, would fit surrogate model and optimize acquisition function
        // For now, return random architectures with bias towards good regions
        // Find best architecture so far
        let best_result = history.iter().max_by(|a, b| {
            let a_score = a.metrics.values().sum::<f64>() / a.metrics.len() as f64;
            let b_score = b.metrics.values().sum::<f64>() / b.metrics.len() as f64;
            a_score.partial_cmp(&b_score).unwrap()
        if let Some(best) = best_result {
            // Generate proposals near the best architecture
                let mutated = best.architecture.mutate(0.1)?;
                proposals.push(Arc::from(mutated));
            // Fallback to random
        // In practice, would update the surrogate model here
        "BayesianOptimization"
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nas::EvaluationMetrics;
    fn create_dummy_result() -> SearchResult {
        let encoding =
            crate::nas::architecture_encoding::SequentialEncoding::random(&mut rng())
                .unwrap();
        let mut metrics = EvaluationMetrics::new();
        metrics.insert("accuracy".to_string(), 0.95);
        SearchResult {
            architecture: Arc::new(encoding),
            metrics,
            training_time: 100.0,
            parameter_count: 1000000,
            flops: Some(1000000),
    #[test]
    fn test_random_search() {
        let search = RandomSearch::new();
        let proposals = search.propose_architectures(&[], 5).unwrap();
        assert_eq!(proposals.len(), 5);
    fn test_evolutionary_search() {
        let mut search = EvolutionarySearch::new(10);
        // Test update
        let results = vec![create_dummy_result(); 5];
        search.update(&results).unwrap();
    fn test_reinforcement_search() {
        let mut search = ReinforcementSearch::new();
        let proposals = search.propose_architectures(&[], 3).unwrap();
        assert_eq!(proposals.len(), 3);
        let results = vec![create_dummy_result(); 3];
        assert!(search.baseline.is_some());
