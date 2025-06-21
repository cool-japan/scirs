# Curriculum Optimization API Reference

The `curriculum_optimization` module provides comprehensive curriculum learning capabilities including task difficulty progression, sample importance weighting, and adversarial training support.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Curriculum Strategies](#curriculum-strategies)
3. [Importance Weighting](#importance-weighting)
4. [Adversarial Training](#adversarial-training)
5. [Adaptive Curriculum](#adaptive-curriculum)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)

## Core Concepts

Curriculum learning involves:
- **Progressive Difficulty**: Starting with easy examples and gradually increasing difficulty
- **Sample Weighting**: Assigning different importance weights to training samples
- **Adversarial Robustness**: Incorporating adversarial examples for robust training
- **Adaptive Scheduling**: Automatically adjusting curriculum based on performance

## Curriculum Strategies

### CurriculumStrategy

Different strategies for curriculum learning progression.

```rust
pub enum CurriculumStrategy {
    /// Linear difficulty progression
    Linear {
        /// Starting difficulty (0.0 to 1.0)
        start_difficulty: f64,
        /// Ending difficulty (0.0 to 1.0)
        end_difficulty: f64,
        /// Number of steps to reach end difficulty
        num_steps: usize,
    },
    /// Exponential difficulty progression
    Exponential {
        /// Starting difficulty (0.0 to 1.0)
        start_difficulty: f64,
        /// Ending difficulty (0.0 to 1.0)
        end_difficulty: f64,
        /// Growth rate
        growth_rate: f64,
    },
    /// Performance-based curriculum
    PerformanceBased {
        /// Threshold for advancing difficulty
        advance_threshold: f64,
        /// Threshold for reducing difficulty
        reduce_threshold: f64,
        /// Difficulty adjustment step size
        adjustment_step: f64,
        /// Window size for performance averaging
        window_size: usize,
    },
    /// Custom curriculum with predefined schedule
    Custom {
        /// Difficulty schedule (step -> difficulty)
        schedule: HashMap<usize, f64>,
        /// Default difficulty for unspecified steps
        default_difficulty: f64,
    },
}
```

### CurriculumManager

Main component for managing curriculum learning.

```rust
pub struct CurriculumManager<A: Float, D: Dimension> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> CurriculumManager<A, D> {
    /// Create a new curriculum manager
    pub fn new(
        strategy: CurriculumStrategy,
        importance_strategy: ImportanceWeightingStrategy,
    ) -> Self;

    /// Enable adversarial training
    pub fn enable_adversarial_training(&mut self, config: AdversarialConfig<A>);

    /// Disable adversarial training
    pub fn disable_adversarial_training(&mut self);

    /// Update curriculum based on performance
    pub fn update_curriculum(&mut self, performance: A) -> Result<()>;

    /// Set difficulty score for a sample
    pub fn set_sample_difficulty(&mut self, sample_id: usize, difficulty: f64);

    /// Check if sample should be included based on current difficulty
    pub fn should_include_sample(&self, sample_id: usize) -> bool;

    /// Get current difficulty level
    pub fn get_current_difficulty(&self) -> f64;

    /// Compute importance weights for samples
    pub fn compute_sample_weights(
        &mut self,
        sample_ids: &[usize],
        losses: &[A],
        gradient_norms: Option<&[A]>,
        uncertainties: Option<&[A]>,
    ) -> Result<()>;

    /// Get importance weight for a sample
    pub fn get_sample_weight(&self, sample_id: usize) -> A;

    /// Generate adversarial examples
    pub fn generate_adversarial_examples(
        &self,
        inputs: &Array<A, D>,
        gradients: &Array<A, D>,
    ) -> Result<Array<A, D>>;

    /// Get filtered samples based on current curriculum
    pub fn filter_samples(&self, sample_ids: &[usize]) -> Vec<usize>;

    /// Reset curriculum state
    pub fn reset(&mut self);

    /// Export curriculum state for analysis
    pub fn export_state(&self) -> CurriculumState<A>;
}
```

## Importance Weighting

### ImportanceWeightingStrategy

Different strategies for weighting training samples.

```rust
pub enum ImportanceWeightingStrategy {
    /// Uniform weighting (all samples equal)
    Uniform,
    /// Loss-based weighting (higher loss = higher weight)
    LossBased {
        /// Temperature parameter for softmax weighting
        temperature: f64,
        /// Minimum weight to avoid zero weights
        min_weight: f64,
    },
    /// Gradient norm based weighting
    GradientNormBased {
        /// Temperature parameter
        temperature: f64,
        /// Minimum weight
        min_weight: f64,
    },
    /// Uncertainty-based weighting
    UncertaintyBased {
        /// Temperature parameter
        temperature: f64,
        /// Minimum weight
        min_weight: f64,
    },
    /// Age-based weighting (older samples get higher weight)
    AgeBased {
        /// Decay factor for age
        decay_factor: f64,
    },
}
```

## Adversarial Training

### AdversarialConfig

Configuration for adversarial training.

```rust
pub struct AdversarialConfig<A: Float> {
    /// Adversarial perturbation magnitude
    pub epsilon: A,
    /// Number of adversarial steps
    pub num_steps: usize,
    /// Step size for adversarial perturbation
    pub step_size: A,
    /// Type of adversarial attack
    pub attack_type: AdversarialAttack,
    /// Regularization weight for adversarial loss
    pub adversarial_weight: A,
}

pub enum AdversarialAttack {
    /// Fast Gradient Sign Method (FGSM)
    FGSM,
    /// Projected Gradient Descent (PGD)
    PGD,
    /// Basic Iterative Method (BIM)
    BIM,
    /// Momentum Iterative Method (MIM)
    MIM,
}
```

## Adaptive Curriculum

### AdaptiveCurriculum

Automatically switches between different curriculum strategies.

```rust
pub struct AdaptiveCurriculum<A: Float, D: Dimension> {
    // Internal fields...
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> AdaptiveCurriculum<A, D> {
    /// Create a new adaptive curriculum
    pub fn new(curricula: Vec<CurriculumManager<A, D>>, switch_threshold: A) -> Self;

    /// Update with performance and potentially switch curriculum
    pub fn update(&mut self, performance: A) -> Result<()>;

    /// Get active curriculum manager
    pub fn active_curriculum(&self) -> &CurriculumManager<A, D>;

    /// Get mutable active curriculum manager
    pub fn active_curriculum_mut(&mut self) -> &mut CurriculumManager<A, D>;

    /// Get active curriculum index
    pub fn active_curriculum_index(&self) -> usize;

    /// Get performance comparison across curricula
    pub fn get_curriculum_comparison(&self) -> Vec<(usize, A)>;
}
```

### CurriculumState

State information for analysis and visualization.

```rust
pub struct CurriculumState<A: Float> {
    /// Current difficulty level
    pub current_difficulty: f64,
    /// Current step count
    pub step_count: usize,
    /// Performance history
    pub performance_history: VecDeque<A>,
    /// Sample weights
    pub sample_weights: HashMap<usize, A>,
    /// Whether adversarial training is enabled
    pub has_adversarial: bool,
}
```

## Usage Examples

### Linear Curriculum Learning

```rust
use scirs2_optim::curriculum_optimization::*;
use ndarray::Array1;

// Create linear curriculum strategy
let strategy = CurriculumStrategy::Linear {
    start_difficulty: 0.1,
    end_difficulty: 1.0,
    num_steps: 1000,
};

let importance_strategy = ImportanceWeightingStrategy::Uniform;
let mut curriculum = CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

// Training loop with curriculum
for step in 0..1000 {
    // Update curriculum based on validation performance
    let val_performance = evaluate_model(&model, &val_data);
    curriculum.update_curriculum(val_performance)?;
    
    // Get current difficulty and filter training samples
    let current_difficulty = curriculum.get_current_difficulty();
    let filtered_samples = curriculum.filter_samples(&training_sample_ids);
    
    // Train on filtered samples
    train_on_samples(&mut model, &filtered_samples, &training_data);
    
    println!("Step {}: difficulty = {:.3}, samples = {}", 
             step, current_difficulty, filtered_samples.len());
}
```

### Performance-Based Curriculum

```rust
// Performance-based curriculum that adapts to learning progress
let strategy = CurriculumStrategy::PerformanceBased {
    advance_threshold: 0.8,   // Increase difficulty when performance > 80%
    reduce_threshold: 0.4,    // Decrease difficulty when performance < 40%
    adjustment_step: 0.1,     // Adjust difficulty by 10% each time
    window_size: 10,          // Average performance over last 10 steps
};

let importance_strategy = ImportanceWeightingStrategy::LossBased {
    temperature: 1.0,
    min_weight: 0.1,
};

let mut curriculum = CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

// Set difficulty scores for samples (e.g., based on data complexity)
for (sample_id, sample) in training_data.iter().enumerate() {
    let difficulty = compute_sample_difficulty(sample); // User-defined function
    curriculum.set_sample_difficulty(sample_id, difficulty);
}

// Training with adaptive curriculum
for epoch in 0..100 {
    let mut epoch_losses = Vec::new();
    
    for batch in batches {
        let (loss, gradients) = forward_backward(&model, &batch);
        epoch_losses.push(loss);
        
        // Update model parameters
        optimizer.step(&gradients, &mut model.parameters());
    }
    
    // Update curriculum based on average loss
    let avg_loss = epoch_losses.iter().sum::<f64>() / epoch_losses.len() as f64;
    let performance = 1.0 / (1.0 + avg_loss); // Convert loss to performance metric
    
    curriculum.update_curriculum(performance)?;
    
    println!("Epoch {}: avg_loss = {:.4}, difficulty = {:.3}", 
             epoch, avg_loss, curriculum.get_current_difficulty());
}
```

### Importance Weighting with Loss-Based Strategy

```rust
// Loss-based importance weighting
let strategy = CurriculumStrategy::Linear {
    start_difficulty: 0.2,
    end_difficulty: 0.9,
    num_steps: 500,
};

let importance_strategy = ImportanceWeightingStrategy::LossBased {
    temperature: 2.0,    // Higher temperature = more uniform weights
    min_weight: 0.05,    // Minimum weight to avoid completely ignoring samples
};

let mut curriculum = CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

// Training loop with importance weighting
for batch in training_batches {
    let (sample_ids, samples, targets) = batch;
    
    // Compute individual sample losses
    let individual_losses = compute_individual_losses(&model, &samples, &targets);
    
    // Compute importance weights based on losses
    curriculum.compute_sample_weights(&sample_ids, &individual_losses, None, None)?;
    
    // Get weights and apply them to loss
    let mut weighted_loss = 0.0;
    for (i, &sample_id) in sample_ids.iter().enumerate() {
        let weight = curriculum.get_sample_weight(sample_id);
        weighted_loss += weight * individual_losses[i];
    }
    weighted_loss /= sample_ids.len() as f64;
    
    // Backward pass with weighted loss
    let gradients = compute_gradients(&model, weighted_loss);
    optimizer.step(&gradients, &mut model.parameters());
}
```

### Adversarial Training Integration

```rust
// Enable adversarial training with FGSM
let adversarial_config = AdversarialConfig {
    epsilon: 0.1,
    num_steps: 1,
    step_size: 0.1,
    attack_type: AdversarialAttack::FGSM,
    adversarial_weight: 0.5,
};

curriculum.enable_adversarial_training(adversarial_config);

// Training with adversarial examples
for batch in training_batches {
    let (samples, targets) = batch;
    
    // Normal forward pass
    let (normal_loss, gradients) = forward_backward(&model, &samples, &targets);
    
    // Generate adversarial examples
    let adversarial_samples = curriculum.generate_adversarial_examples(&samples, &gradients)?;
    
    // Forward pass on adversarial examples
    let (adv_loss, adv_gradients) = forward_backward(&model, &adversarial_samples, &targets);
    
    // Combine losses
    let combined_loss = 0.5 * normal_loss + 0.5 * adv_loss;
    let combined_gradients = combine_gradients(&gradients, &adv_gradients, 0.5);
    
    // Update model
    optimizer.step(&combined_gradients, &mut model.parameters());
    
    println!("Normal loss: {:.4}, Adversarial loss: {:.4}", normal_loss, adv_loss);
}
```

### PGD Adversarial Training

```rust
// More sophisticated adversarial training with PGD
let adversarial_config = AdversarialConfig {
    epsilon: 0.03,
    num_steps: 10,
    step_size: 0.007,
    attack_type: AdversarialAttack::PGD,
    adversarial_weight: 0.6,
};

curriculum.enable_adversarial_training(adversarial_config);

// Multi-step adversarial training
for batch in training_batches {
    let (samples, targets) = batch;
    
    // Generate strong adversarial examples with PGD
    let gradients = compute_gradients_wrt_inputs(&model, &samples, &targets);
    let adversarial_samples = curriculum.generate_adversarial_examples(&samples, &gradients)?;
    
    // Train on both clean and adversarial examples
    let clean_loss = compute_loss(&model, &samples, &targets);
    let adv_loss = compute_loss(&model, &adversarial_samples, &targets);
    
    let total_loss = (1.0 - 0.6) * clean_loss + 0.6 * adv_loss;
    
    // Update model to be robust to adversarial examples
    let model_gradients = compute_gradients(&model, total_loss);
    optimizer.step(&model_gradients, &mut model.parameters());
}
```

### Adaptive Curriculum Management

```rust
// Create multiple curriculum strategies
let linear_strategy = CurriculumStrategy::Linear {
    start_difficulty: 0.1,
    end_difficulty: 0.8,
    num_steps: 1000,
};

let exponential_strategy = CurriculumStrategy::Exponential {
    start_difficulty: 0.2,
    end_difficulty: 0.9,
    growth_rate: 0.01,
};

let performance_strategy = CurriculumStrategy::PerformanceBased {
    advance_threshold: 0.75,
    reduce_threshold: 0.45,
    adjustment_step: 0.05,
    window_size: 20,
};

// Create curriculum managers
let importance_strategy = ImportanceWeightingStrategy::Uniform;
let curriculum1 = CurriculumManager::<f64, ndarray::Ix1>::new(linear_strategy, importance_strategy.clone());
let curriculum2 = CurriculumManager::<f64, ndarray::Ix1>::new(exponential_strategy, importance_strategy.clone());
let curriculum3 = CurriculumManager::<f64, ndarray::Ix1>::new(performance_strategy, importance_strategy);

// Create adaptive curriculum
let mut adaptive = AdaptiveCurriculum::new(
    vec![curriculum1, curriculum2, curriculum3],
    0.05 // Switch threshold
);

// Training with adaptive curriculum selection
for epoch in 0..200 {
    let performance = train_one_epoch(&mut model, &adaptive)?;
    
    // Update adaptive curriculum
    adaptive.update(performance)?;
    
    // Get current active curriculum
    let active_idx = adaptive.active_curriculum_index();
    let current_difficulty = adaptive.active_curriculum().get_current_difficulty();
    
    println!("Epoch {}: active_curriculum = {}, difficulty = {:.3}, performance = {:.4}",
             epoch, active_idx, current_difficulty, performance);
    
    // Analyze curriculum performance
    if epoch % 50 == 0 {
        let comparison = adaptive.get_curriculum_comparison();
        println!("Curriculum performance comparison: {:?}", comparison);
    }
}
```

### Uncertainty-Based Importance Weighting

```rust
// Use model uncertainty for importance weighting
let importance_strategy = ImportanceWeightingStrategy::UncertaintyBased {
    temperature: 1.5,
    min_weight: 0.1,
};

let mut curriculum = CurriculumManager::<f64, ndarray::Ix1>::new(
    CurriculumStrategy::Linear {
        start_difficulty: 0.3,
        end_difficulty: 1.0,
        num_steps: 800,
    },
    importance_strategy,
);

// Training with uncertainty-based weighting
for batch in training_batches {
    let (sample_ids, samples, targets) = batch;
    
    // Compute model predictions and uncertainties
    let (predictions, uncertainties) = model.predict_with_uncertainty(&samples);
    let losses = compute_losses(&predictions, &targets);
    
    // Compute importance weights based on uncertainties
    curriculum.compute_sample_weights(
        &sample_ids,
        &losses,
        None,
        Some(&uncertainties),
    )?;
    
    // Apply weights to training
    let mut weighted_gradients = Vec::new();
    for (i, &sample_id) in sample_ids.iter().enumerate() {
        let weight = curriculum.get_sample_weight(sample_id);
        let sample_gradients = compute_sample_gradients(&model, &samples[i], &targets[i]);
        
        // Scale gradients by importance weight
        let weighted_sample_gradients = scale_gradients(&sample_gradients, weight);
        weighted_gradients.push(weighted_sample_gradients);
    }
    
    // Average weighted gradients
    let avg_gradients = average_gradients(&weighted_gradients);
    optimizer.step(&avg_gradients, &mut model.parameters());
}
```

### Custom Curriculum Schedule

```rust
// Define custom curriculum schedule
let mut schedule = HashMap::new();
schedule.insert(0, 0.1);     // Start easy
schedule.insert(100, 0.3);   // Ramp up
schedule.insert(300, 0.6);   // Medium difficulty
schedule.insert(600, 0.8);   // Hard
schedule.insert(900, 1.0);   // Full difficulty

let strategy = CurriculumStrategy::Custom {
    schedule,
    default_difficulty: 1.0,
};

let importance_strategy = ImportanceWeightingStrategy::AgeBased {
    decay_factor: 0.01,
};

let mut curriculum = CurriculumManager::<f64, ndarray::Ix1>::new(strategy, importance_strategy);

// Training with custom schedule
for step in 0..1000 {
    // Curriculum automatically follows the predefined schedule
    curriculum.update_curriculum(0.5)?; // Dummy performance (not used for custom schedule)
    
    let difficulty = curriculum.get_current_difficulty();
    
    // Filter samples based on current difficulty
    let filtered_samples = curriculum.filter_samples(&all_sample_ids);
    
    // Age-based weighting gives higher weight to older samples
    let sample_ages: Vec<f64> = filtered_samples
        .iter()
        .map(|&id| (step - id) as f64)
        .collect();
    
    // Training step
    train_on_filtered_samples(&mut model, &filtered_samples, step);
    
    if step % 100 == 0 {
        println!("Step {}: difficulty = {:.1}, active_samples = {}", 
                 step, difficulty, filtered_samples.len());
    }
}
```

## Best Practices

### Curriculum Design

1. **Start Simple**: Begin with easy examples that the model can learn quickly
2. **Gradual Progression**: Increase difficulty gradually, not abruptly
3. **Performance Monitoring**: Adjust curriculum based on learning progress
4. **Domain Knowledge**: Use task-specific knowledge to define difficulty
5. **Validation**: Regularly evaluate on held-out data

### Sample Weighting

1. **Balance Exploration/Exploitation**: Don't completely ignore any samples
2. **Temperature Tuning**: Adjust temperature to control weight distribution
3. **Minimum Weights**: Use minimum weights to ensure all samples contribute
4. **Normalization**: Ensure weights are properly normalized
5. **Stability**: Avoid rapidly changing weights that could destabilize training

### Adversarial Training

1. **Budget Allocation**: Balance clean and adversarial examples
2. **Attack Strength**: Start with weak attacks and gradually increase
3. **Diversity**: Use multiple attack types for robust training
4. **Computational Cost**: Consider the computational overhead
5. **Evaluation**: Test robustness on various attack methods

### Implementation Tips

```rust
// Monitor curriculum progression
let state = curriculum.export_state();
log_curriculum_state(&state, step);

// Save curriculum state for reproducibility
let checkpoint = CurriculumCheckpoint {
    state: curriculum.export_state(),
    random_seed: get_random_seed(),
    step: current_step,
};
save_checkpoint("curriculum_checkpoint.json", &checkpoint)?;

// Visualize curriculum progress
if step % 100 == 0 {
    plot_difficulty_progression(&curriculum.get_performance_history());
    plot_sample_weight_distribution(&curriculum.export_state().sample_weights);
}

// Handle edge cases
if curriculum.get_current_difficulty() < 0.01 {
    // Model is struggling, slow down curriculum
    curriculum.update_curriculum(0.3)?; // Force lower performance feedback
}

// Combine with other techniques
let regularization_strength = 1.0 - curriculum.get_current_difficulty();
apply_regularization(&mut model, regularization_strength);
```

The curriculum optimization API provides comprehensive tools for implementing sophisticated curriculum learning strategies that can significantly improve training efficiency and model performance.