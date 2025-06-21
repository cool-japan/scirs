# Fine-tuning Pre-trained Models Tutorial

This comprehensive tutorial covers how to fine-tune pre-trained neural network models using scirs2-neural, from basic transfer learning to advanced fine-tuning strategies.

## Table of Contents

1. [Introduction to Fine-tuning](#introduction-to-fine-tuning)
2. [Setting Up for Fine-tuning](#setting-up-for-fine-tuning)
3. [Loading Pre-trained Models](#loading-pre-trained-models)
4. [Basic Transfer Learning](#basic-transfer-learning)
5. [Advanced Fine-tuning Strategies](#advanced-fine-tuning-strategies)
6. [Domain Adaptation](#domain-adaptation)
7. [Few-shot Learning](#few-shot-learning)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction to Fine-tuning

Fine-tuning is the process of adapting a pre-trained model to a new task or domain by continuing training on task-specific data. This approach leverages the knowledge learned from large datasets and can achieve better performance with less data and computational resources.

### When to Use Fine-tuning

- **Limited training data**: When you have insufficient data to train from scratch
- **Similar domains**: When your task is related to the pre-training domain
- **Resource constraints**: When you have limited computational resources
- **Quick prototyping**: When you need to quickly test model performance

### Types of Fine-tuning

1. **Feature Extraction**: Freeze pre-trained layers and train only new layers
2. **Fine-tuning**: Unfreeze some or all layers and train with lower learning rates
3. **Domain Adaptation**: Adapt model to new domains while preserving general knowledge
4. **Few-shot Learning**: Learn new tasks with very few examples

## Setting Up for Fine-tuning

### Dependencies and Imports

```rust
use scirs2_neural::prelude::*;
use scirs2_neural::layers::{Sequential, Dense, Conv2D, PaddingMode, MaxPool2D, Dropout, BatchNorm};
use scirs2_neural::training::{Trainer, TrainingConfig, ValidationSettings};
use scirs2_neural::losses::{CrossEntropyLoss, MeanSquaredError};
use scirs2_neural::serialization::{ModelSerializer, SerializationFormat};
use scirs2_neural::transfer::{TransferLearning, FreezingStrategy, LayerSelector};
use ndarray::{Array, ArrayD, IxDyn};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::path::Path;
```

### Configuration for Fine-tuning

```rust
#[derive(Debug, Clone)]
pub struct FinetuningConfig {
    pub pretrained_model_path: String,
    pub target_classes: usize,
    pub freezing_strategy: FreezingStrategy,
    pub learning_rate: f64,
    pub fine_tuning_epochs: usize,
    pub warm_up_epochs: usize,
    pub layer_wise_lr: bool,
}

impl Default for FinetuningConfig {
    fn default() -> Self {
        Self {
            pretrained_model_path: "models/pretrained_model.safetensors".to_string(),
            target_classes: 10,
            freezing_strategy: FreezingStrategy::PartialFreeze(0.7),
            learning_rate: 1e-4, // Lower learning rate for fine-tuning
            fine_tuning_epochs: 20,
            warm_up_epochs: 5,
            layer_wise_lr: true,
        }
    }
}
```

## Loading Pre-trained Models

### Basic Model Loading

```rust
/// Load a pre-trained model from disk
fn load_pretrained_model(model_path: &str) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let serializer = ModelSerializer::new();
    let model = serializer.load_model(model_path, SerializationFormat::SafeTensors)?;
    
    println!("✅ Loaded pre-trained model from: {}", model_path);
    println!("📊 Model summary:");
    println!("   - Total parameters: {}", model.parameter_count());
    println!("   - Number of layers: {}", model.layer_count());
    
    Ok(model)
}

/// Inspect model architecture
fn inspect_model_architecture(model: &Sequential<f32>) {
    println!("\n🏗️ Model Architecture:");
    for (i, layer) in model.layers().iter().enumerate() {
        println!("   Layer {}: {}", i, layer.layer_description());
    }
}
```

### Model Compatibility Checking

```rust
/// Check if pre-trained model is compatible with target task
fn check_model_compatibility(
    model: &Sequential<f32>,
    input_shape: &[usize],
    target_classes: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Check input compatibility
    let dummy_input = Array::ones(IxDyn(input_shape));
    let output = model.forward(&dummy_input)?;
    
    println!("🔍 Compatibility Check:");
    println!("   - Input shape: {:?}", input_shape);
    println!("   - Output shape: {:?}", output.shape());
    println!("   - Expected classes: {}", target_classes);
    
    // Check if output dimension matches or can be adapted
    let output_dim = output.shape().last().copied().unwrap_or(0);
    let compatible = output_dim == target_classes || output_dim > target_classes;
    
    if compatible {
        println!("   ✅ Model is compatible");
    } else {
        println!("   ⚠️ Model requires adaptation");
    }
    
    Ok(compatible)
}
```

## Basic Transfer Learning

### Feature Extraction Approach

```rust
/// Create a feature extraction model by freezing pre-trained layers
fn create_feature_extractor(
    pretrained_model: Sequential<f32>,
    target_classes: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let mut model = pretrained_model;
    
    // Freeze all layers except the last few
    let total_layers = model.layer_count();
    let freeze_until = (total_layers as f32 * 0.8) as usize; // Freeze 80% of layers
    
    for i in 0..freeze_until {
        if let Some(layer) = model.get_layer_mut(i) {
            layer.set_trainable(false);
            println!("❄️ Frozen layer {}: {}", i, layer.layer_type());
        }
    }
    
    // Replace the final classification layer
    model.pop_layer(); // Remove old classification layer
    model.add(Dense::new(
        model.get_output_dim(),
        target_classes,
        Some("softmax"),
        rng,
    )?);
    
    println!("🔧 Added new classification head for {} classes", target_classes);
    
    Ok(model)
}
```

### Training with Feature Extraction

```rust
fn train_feature_extractor(
    model: Sequential<f32>,
    train_data: &[(ArrayD<f32>, usize)],
    val_data: &[(ArrayD<f32>, usize)],
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let config = TrainingConfig {
        batch_size: 32,
        epochs: 10,
        learning_rate: 1e-3, // Higher learning rate for new layers
        shuffle: true,
        verbose: 1,
        validation: Some(ValidationSettings {
            enabled: true,
            validation_split: 0.0, // Using separate validation data
            batch_size: 64,
            num_workers: 0,
        }),
        gradient_accumulation: None,
        mixed_precision: None,
    };
    
    let loss_fn = CrossEntropyLoss::new();
    let optimizer = Adam::new(config.learning_rate as f32);
    
    println!("🚀 Starting feature extraction training...");
    let mut trainer = Trainer::new(model, optimizer, loss_fn, config);
    
    // Train only the new layers
    for epoch in 0..10 {
        println!("\n📈 Epoch {}/10", epoch + 1);
        
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        
        for (batch_idx, (inputs, targets)) in train_data.iter().enumerate() {
            let outputs = trainer.model.forward(inputs)?;
            
            // Compute accuracy
            let predicted = argmax(&outputs);
            if predicted == *targets {
                correct += 1;
            }
            total += 1;
            
            if batch_idx % 100 == 0 {
                print!("🔄 Batch {} - Accuracy: {:.2}%\r", 
                       batch_idx, (correct as f32 / total as f32) * 100.0);
            }
        }
        
        let accuracy = (correct as f32 / total as f32) * 100.0;
        println!("✅ Epoch {} - Training Accuracy: {:.2}%", epoch + 1, accuracy);
        
        // Validation
        if epoch % 2 == 0 {
            let val_accuracy = evaluate_model(&trainer.model, val_data)?;
            println!("📊 Validation Accuracy: {:.2}%", val_accuracy);
        }
    }
    
    println!("🎉 Feature extraction training completed!");
    Ok(trainer.model)
}
```

## Advanced Fine-tuning Strategies

### Layer-wise Learning Rate Scheduling

```rust
/// Apply different learning rates to different layers
fn apply_layer_wise_learning_rates(
    model: &mut Sequential<f32>,
    base_lr: f32,
    decay_factor: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_layers = model.layer_count();
    
    println!("🎯 Applying layer-wise learning rates:");
    
    for i in 0..total_layers {
        if let Some(layer) = model.get_layer_mut(i) {
            // Earlier layers get lower learning rates
            let layer_lr = base_lr * decay_factor.powi(total_layers as i32 - i as i32 - 1);
            layer.set_learning_rate(layer_lr);
            
            println!("   Layer {}: lr = {:.2e}", i, layer_lr);
        }
    }
    
    Ok(())
}
```

### Gradual Unfreezing

```rust
/// Gradually unfreeze layers during training
fn gradual_unfreezing_training(
    mut model: Sequential<f32>,
    train_data: &[(ArrayD<f32>, usize)],
    config: &FinetuningConfig,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    let total_layers = model.layer_count();
    let unfreeze_interval = config.fine_tuning_epochs / 4; // Unfreeze in 4 stages
    
    for stage in 0..4 {
        println!("\n🔓 Stage {}: Unfreezing layers...", stage + 1);
        
        // Unfreeze more layers in each stage
        let layers_to_unfreeze = (total_layers / 4) * (stage + 1);
        for i in (total_layers - layers_to_unfreeze)..total_layers {
            if let Some(layer) = model.get_layer_mut(i) {
                layer.set_trainable(true);
                println!("   🔥 Unfrozen layer {}: {}", i, layer.layer_type());
            }
        }
        
        // Train for a few epochs with current configuration
        for epoch in 0..unfreeze_interval {
            let global_epoch = stage * unfreeze_interval + epoch + 1;
            println!("📈 Global Epoch {}/{}", global_epoch, config.fine_tuning_epochs);
            
            // Reduce learning rate as we unfreeze more layers
            let current_lr = config.learning_rate * 0.5_f64.powi(stage as i32);
            
            // Training step (simplified)
            train_epoch(&mut model, train_data, current_lr as f32)?;
        }
    }
    
    Ok(model)
}
```

### Discriminative Fine-tuning

```rust
/// Apply discriminative fine-tuning with different learning rates per layer group
fn discriminative_fine_tuning(
    mut model: Sequential<f32>,
    train_data: &[(ArrayD<f32>, usize)],
    config: &FinetuningConfig,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    println!("🎯 Starting discriminative fine-tuning...");
    
    let total_layers = model.layer_count();
    
    // Define layer groups with different learning rates
    let groups = vec![
        LayerGroup { start: 0, end: total_layers / 3, lr_multiplier: 0.1 },
        LayerGroup { start: total_layers / 3, end: 2 * total_layers / 3, lr_multiplier: 0.3 },
        LayerGroup { start: 2 * total_layers / 3, end: total_layers, lr_multiplier: 1.0 },
    ];
    
    // Apply different learning rates to each group
    for (group_idx, group) in groups.iter().enumerate() {
        let group_lr = config.learning_rate * group.lr_multiplier as f64;
        
        println!("👥 Group {} (layers {}-{}): lr = {:.2e}", 
                 group_idx + 1, group.start, group.end - 1, group_lr);
        
        for i in group.start..group.end {
            if let Some(layer) = model.get_layer_mut(i) {
                layer.set_learning_rate(group_lr as f32);
            }
        }
    }
    
    // Training loop with adaptive learning rate scheduling
    for epoch in 0..config.fine_tuning_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, config.fine_tuning_epochs);
        
        // Cosine annealing learning rate schedule
        let lr_scale = 0.5 * (1.0 + ((epoch as f64 * std::f64::consts::PI) / 
                                    config.fine_tuning_epochs as f64).cos());
        
        // Apply learning rate scaling to all groups
        for (group_idx, group) in groups.iter().enumerate() {
            let scaled_lr = config.learning_rate * group.lr_multiplier as f64 * lr_scale;
            
            for i in group.start..group.end {
                if let Some(layer) = model.get_layer_mut(i) {
                    layer.set_learning_rate(scaled_lr as f32);
                }
            }
        }
        
        // Training step
        let epoch_loss = train_epoch(&mut model, train_data, config.learning_rate as f32)?;
        println!("✅ Epoch {} completed - Loss: {:.4}", epoch + 1, epoch_loss);
        
        // Early stopping check
        if epoch_loss < 0.01 {
            println!("🎯 Early stopping - Loss threshold reached");
            break;
        }
    }
    
    Ok(model)
}

#[derive(Debug)]
struct LayerGroup {
    start: usize,
    end: usize,
    lr_multiplier: f32,
}
```

## Domain Adaptation

### Domain Adversarial Training

```rust
/// Implement domain adversarial training for domain adaptation
fn domain_adversarial_training(
    mut feature_extractor: Sequential<f32>,
    mut classifier: Sequential<f32>,
    mut domain_discriminator: Sequential<f32>,
    source_data: &[(ArrayD<f32>, usize)],
    target_data: &[(ArrayD<f32>, usize)],
) -> Result<(Sequential<f32>, Sequential<f32>), Box<dyn std::error::Error>> {
    println!("🎭 Starting domain adversarial training...");
    
    let num_epochs = 30;
    let lambda = 0.1; // Trade-off parameter
    
    for epoch in 0..num_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, num_epochs);
        
        let mut classification_loss = 0.0;
        let mut domain_loss = 0.0;
        
        // Train on source domain with classification loss
        for (inputs, labels) in source_data.iter().take(100) {
            // Forward pass through feature extractor
            let features = feature_extractor.forward(inputs)?;
            
            // Classification loss
            let class_output = classifier.forward(&features)?;
            let class_loss = compute_classification_loss(&class_output, *labels)?;
            classification_loss += class_loss;
            
            // Domain discrimination (label = 0 for source)
            let domain_output = domain_discriminator.forward(&features)?;
            let domain_class_loss = compute_domain_loss(&domain_output, 0)?;
            domain_loss += domain_class_loss;
        }
        
        // Train on target domain with domain confusion loss
        for (inputs, _) in target_data.iter().take(100) {
            let features = feature_extractor.forward(inputs)?;
            
            // Domain discrimination (label = 1 for target)
            let domain_output = domain_discriminator.forward(&features)?;
            let domain_class_loss = compute_domain_loss(&domain_output, 1)?;
            domain_loss += domain_class_loss;
        }
        
        // Gradient reversal for domain confusion
        let total_loss = classification_loss - lambda * domain_loss;
        
        println!("   📊 Classification Loss: {:.4}", classification_loss / 100.0);
        println!("   🎭 Domain Loss: {:.4}", domain_loss / 200.0);
        println!("   📈 Total Loss: {:.4}", total_loss / 100.0);
        
        // Update learning rate schedule
        if epoch % 10 == 0 && epoch > 0 {
            let new_lr = 0.001 * 0.5_f32.powi(epoch / 10);
            println!("   📉 Learning rate reduced to: {:.2e}", new_lr);
        }
    }
    
    println!("🎉 Domain adversarial training completed!");
    Ok((feature_extractor, classifier))
}
```

### Progressive Domain Adaptation

```rust
/// Progressive domain adaptation with curriculum learning
fn progressive_domain_adaptation(
    mut model: Sequential<f32>,
    source_data: &[(ArrayD<f32>, usize)],
    target_data: &[(ArrayD<f32>, usize)],
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    println!("🎯 Starting progressive domain adaptation...");
    
    let num_stages = 5;
    let epochs_per_stage = 10;
    
    for stage in 0..num_stages {
        println!("\n🎪 Stage {}/{}", stage + 1, num_stages);
        
        // Gradually increase target domain ratio
        let target_ratio = (stage + 1) as f32 / num_stages as f32;
        let source_ratio = 1.0 - target_ratio;
        
        println!("   📊 Source ratio: {:.1}%, Target ratio: {:.1}%", 
                 source_ratio * 100.0, target_ratio * 100.0);
        
        for epoch in 0..epochs_per_stage {
            let global_epoch = stage * epochs_per_stage + epoch + 1;
            println!("   📈 Stage {}, Epoch {}/{}", stage + 1, epoch + 1, epochs_per_stage);
            
            // Mix source and target data according to current ratios
            let mixed_data = create_mixed_batch(source_data, target_data, source_ratio, target_ratio)?;
            
            // Train on mixed data
            let epoch_loss = train_epoch(&mut model, &mixed_data, 0.001)?;
            
            if epoch % 5 == 0 {
                println!("     💫 Loss: {:.4}", epoch_loss);
            }
        }
        
        // Evaluate adaptation progress
        let source_acc = evaluate_model(&model, &source_data[..100])?;
        let target_acc = evaluate_model(&model, &target_data[..100])?;
        
        println!("   📊 Source accuracy: {:.2}%", source_acc);
        println!("   🎯 Target accuracy: {:.2}%", target_acc);
    }
    
    Ok(model)
}
```

## Few-shot Learning

### Prototypical Networks

```rust
/// Implement prototypical networks for few-shot learning
fn prototypical_network_training(
    mut feature_extractor: Sequential<f32>,
    support_set: &[(ArrayD<f32>, usize)],
    query_set: &[(ArrayD<f32>, usize)],
    n_way: usize,
    k_shot: usize,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    println!("🎯 Training prototypical network: {}-way {}-shot", n_way, k_shot);
    
    let num_episodes = 1000;
    
    for episode in 0..num_episodes {
        if episode % 100 == 0 {
            println!("📈 Episode {}/{}", episode + 1, num_episodes);
        }
        
        // Sample support set
        let sampled_support = sample_support_set(support_set, n_way, k_shot)?;
        
        // Compute prototypes for each class
        let mut prototypes = Vec::new();
        for class_id in 0..n_way {
            let class_samples: Vec<_> = sampled_support.iter()
                .filter(|(_, label)| *label == class_id)
                .collect();
            
            if !class_samples.is_empty() {
                let prototype = compute_prototype(&feature_extractor, &class_samples)?;
                prototypes.push(prototype);
            }
        }
        
        // Evaluate on query set
        let mut episode_loss = 0.0;
        let mut correct = 0;
        
        for (query_input, query_label) in query_set.iter().take(20) {
            let query_features = feature_extractor.forward(query_input)?;
            
            // Find nearest prototype
            let (predicted_class, confidence) = classify_by_prototype(&query_features, &prototypes)?;
            
            if predicted_class == *query_label {
                correct += 1;
            }
            
            // Compute prototypical loss
            let loss = prototypical_loss(&query_features, &prototypes, *query_label)?;
            episode_loss += loss;
        }
        
        let accuracy = correct as f32 / 20.0 * 100.0;
        
        if episode % 100 == 0 {
            println!("   📊 Episode accuracy: {:.2}%", accuracy);
            println!("   📉 Episode loss: {:.4}", episode_loss / 20.0);
        }
        
        // Update model based on prototypical loss
        // (In practice, you would compute gradients and update parameters)
    }
    
    println!("🎉 Prototypical network training completed!");
    Ok(feature_extractor)
}
```

### Meta-learning (MAML-style)

```rust
/// Model-Agnostic Meta-Learning for few-shot adaptation
fn meta_learning_training(
    mut model: Sequential<f32>,
    task_distribution: &[Vec<(ArrayD<f32>, usize)>],
    inner_lr: f32,
    outer_lr: f32,
    inner_steps: usize,
) -> Result<Sequential<f32>, Box<dyn std::error::Error>> {
    println!("🧠 Starting meta-learning training (MAML-style)");
    
    let num_meta_epochs = 100;
    
    for meta_epoch in 0..num_meta_epochs {
        println!("🔄 Meta-epoch {}/{}", meta_epoch + 1, num_meta_epochs);
        
        let mut meta_loss = 0.0;
        let batch_size = 5; // Number of tasks per meta-update
        
        for task_batch in task_distribution.chunks(batch_size) {
            let mut task_losses = Vec::new();
            
            for task in task_batch {
                // Split task into support and query sets
                let (support_set, query_set) = split_task_data(task, 0.5)?;
                
                // Clone model for inner loop
                let mut task_model = model.clone();
                
                // Inner loop: adapt to task
                for inner_step in 0..inner_steps {
                    let support_loss = train_epoch(&mut task_model, &support_set, inner_lr)?;
                    
                    if inner_step == 0 || inner_step == inner_steps - 1 {
                        println!("   🔹 Task inner step {}: loss = {:.4}", inner_step + 1, support_loss);
                    }
                }
                
                // Evaluate adapted model on query set
                let query_loss = evaluate_loss(&task_model, &query_set)?;
                task_losses.push(query_loss);
                
                println!("   📊 Task query loss: {:.4}", query_loss);
            }
            
            // Meta-update: update original model based on query losses
            let avg_task_loss = task_losses.iter().sum::<f32>() / task_losses.len() as f32;
            meta_loss += avg_task_loss;
            
            // (In practice, compute meta-gradients and update model parameters)
            update_meta_parameters(&mut model, outer_lr, avg_task_loss)?;
        }
        
        let avg_meta_loss = meta_loss / (task_distribution.len() / batch_size) as f32;
        println!("✅ Meta-epoch {} completed - Meta-loss: {:.4}", meta_epoch + 1, avg_meta_loss);
        
        // Decay learning rates
        if meta_epoch % 20 == 0 && meta_epoch > 0 {
            let decay_factor = 0.8;
            println!("📉 Decaying learning rates by {:.2}", decay_factor);
        }
    }
    
    println!("🎉 Meta-learning training completed!");
    Ok(model)
}
```

## Best Practices

### 1. Learning Rate Scheduling

```rust
/// Implement cosine annealing with warm restarts
fn cosine_annealing_lr(
    initial_lr: f32,
    current_epoch: usize,
    total_epochs: usize,
    restart_period: usize,
) -> f32 {
    let cycle_epoch = current_epoch % restart_period;
    let cycle_progress = cycle_epoch as f32 / restart_period as f32;
    
    initial_lr * 0.5 * (1.0 + (std::f32::consts::PI * cycle_progress).cos())
}

/// Apply learning rate warmup
fn warmup_lr(current_epoch: usize, warmup_epochs: usize, target_lr: f32) -> f32 {
    if current_epoch < warmup_epochs {
        target_lr * (current_epoch + 1) as f32 / warmup_epochs as f32
    } else {
        target_lr
    }
}
```

### 2. Data Augmentation for Fine-tuning

```rust
/// Apply task-specific data augmentation
fn apply_fine_tuning_augmentation(
    input: &ArrayD<f32>,
    augmentation_strength: f32,
) -> Result<ArrayD<f32>, Box<dyn std::error::Error>> {
    let mut augmented = input.clone();
    
    // Light augmentation for fine-tuning (preserve pre-trained features)
    if augmentation_strength > 0.5 {
        // Horizontal flip (50% chance)
        if rand::random::<f32>() < 0.5 {
            augmented = horizontal_flip(&augmented)?;
        }
        
        // Small rotation (-5 to 5 degrees)
        let angle = (rand::random::<f32>() - 0.5) * 10.0 * augmentation_strength;
        augmented = rotate(&augmented, angle)?;
        
        // Slight color jittering
        let color_jitter = 0.1 * augmentation_strength;
        augmented = color_jitter(&augmented, color_jitter)?;
    }
    
    // Mild noise (always applied)
    let noise_level = 0.01 * augmentation_strength;
    augmented = add_gaussian_noise(&augmented, noise_level)?;
    
    Ok(augmented)
}
```

### 3. Model Evaluation and Monitoring

```rust
/// Comprehensive evaluation during fine-tuning
fn evaluate_fine_tuning_progress(
    model: &Sequential<f32>,
    val_data: &[(ArrayD<f32>, usize)],
    original_task_data: &[(ArrayD<f32>, usize)],
    epoch: usize,
) -> Result<EvaluationMetrics, Box<dyn std::error::Error>> {
    println!("🔍 Evaluating fine-tuning progress...");
    
    // Target task performance
    let target_accuracy = evaluate_model(model, val_data)?;
    
    // Original task performance (catastrophic forgetting check)
    let original_accuracy = evaluate_model(model, original_task_data)?;
    
    // Feature similarity analysis
    let feature_similarity = compute_feature_similarity(model, val_data, original_task_data)?;
    
    let metrics = EvaluationMetrics {
        epoch,
        target_accuracy,
        original_accuracy,
        feature_similarity,
        forgetting_measure: calculate_forgetting_measure(target_accuracy, original_accuracy),
    };
    
    println!("📊 Evaluation Results:");
    println!("   🎯 Target task accuracy: {:.2}%", metrics.target_accuracy);
    println!("   🔄 Original task accuracy: {:.2}%", metrics.original_accuracy);
    println!("   📐 Feature similarity: {:.4}", metrics.feature_similarity);
    println!("   🧠 Forgetting measure: {:.4}", metrics.forgetting_measure);
    
    // Early stopping criteria
    if metrics.forgetting_measure > 0.3 {
        println!("⚠️ Warning: High catastrophic forgetting detected!");
    }
    
    if metrics.target_accuracy > 95.0 {
        println!("🎉 Excellent target performance achieved!");
    }
    
    Ok(metrics)
}

#[derive(Debug)]
struct EvaluationMetrics {
    epoch: usize,
    target_accuracy: f32,
    original_accuracy: f32,
    feature_similarity: f32,
    forgetting_measure: f32,
}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Catastrophic Forgetting

```rust
/// Detect and mitigate catastrophic forgetting
fn mitigate_catastrophic_forgetting(
    model: &mut Sequential<f32>,
    original_data: &[(ArrayD<f32>, usize)],
    target_data: &[(ArrayD<f32>, usize)],
    regularization_strength: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ Applying catastrophic forgetting mitigation...");
    
    // Elastic Weight Consolidation (EWC)
    let fisher_information = compute_fisher_information(model, original_data)?;
    
    // Add EWC regularization to loss
    for (layer_idx, layer) in model.layers_mut().iter_mut().enumerate() {
        if let Some(params) = layer.get_parameters_mut() {
            let fisher_values = &fisher_information[layer_idx];
            layer.set_ewc_regularization(fisher_values, regularization_strength);
            
            println!("   🔒 Applied EWC to layer {}", layer_idx);
        }
    }
    
    // Memory replay with original samples
    let replay_ratio = 0.2; // 20% of batch from original task
    implement_memory_replay(model, original_data, replay_ratio)?;
    
    println!("✅ Catastrophic forgetting mitigation applied");
    Ok(())
}
```

#### 2. Overfitting

```rust
/// Prevent overfitting during fine-tuning
fn prevent_overfitting(
    model: &mut Sequential<f32>,
    config: &mut FinetuningConfig,
    val_performance: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    // Check for overfitting signs
    let recent_performance: Vec<f32> = val_performance.iter()
        .rev()
        .take(5)
        .cloned()
        .collect();
    
    let is_overfitting = recent_performance.len() >= 3 && 
        recent_performance[0] < recent_performance[2];
    
    if is_overfitting {
        println!("⚠️ Overfitting detected - applying countermeasures");
        
        // Reduce learning rate
        config.learning_rate *= 0.5;
        println!("   📉 Learning rate reduced to: {:.2e}", config.learning_rate);
        
        // Increase dropout
        for layer in model.layers_mut() {
            if layer.layer_type() == "Dropout" {
                layer.increase_dropout_rate(0.1);
                println!("   🎯 Increased dropout rate");
            }
        }
        
        // Add weight decay
        for layer in model.layers_mut() {
            if layer.has_parameters() {
                layer.set_weight_decay(0.01);
                println!("   ⚖️ Applied weight decay");
            }
        }
        
        // Early stopping
        if recent_performance.len() >= 5 {
            let trend = recent_performance[0] - recent_performance[4];
            if trend < -0.02 {
                println!("   🛑 Suggesting early stopping");
                return Err("Early stopping recommended".into());
            }
        }
    }
    
    Ok(())
}
```

#### 3. Poor Transfer Performance

```rust
/// Debug poor transfer performance
fn debug_transfer_performance(
    model: &Sequential<f32>,
    source_data: &[(ArrayD<f32>, usize)],
    target_data: &[(ArrayD<f32>, usize)],
) -> Result<TransferDiagnostics, Box<dyn std::error::Error>> {
    println!("🔍 Diagnosing transfer performance...");
    
    // Feature analysis
    let source_features = extract_features(model, source_data)?;
    let target_features = extract_features(model, target_data)?;
    
    // Domain shift measurement
    let domain_shift = measure_domain_shift(&source_features, &target_features)?;
    
    // Layer-wise transferability
    let transferability_scores = compute_layer_transferability(model, source_data, target_data)?;
    
    let diagnostics = TransferDiagnostics {
        domain_shift,
        transferability_scores,
        recommended_strategy: recommend_strategy(domain_shift, &transferability_scores),
    };
    
    println!("📊 Transfer Diagnostics:");
    println!("   🔄 Domain shift: {:.4}", diagnostics.domain_shift);
    println!("   📈 Average transferability: {:.4}", 
             diagnostics.transferability_scores.iter().sum::<f32>() / 
             diagnostics.transferability_scores.len() as f32);
    println!("   💡 Recommended strategy: {:?}", diagnostics.recommended_strategy);
    
    Ok(diagnostics)
}

#[derive(Debug)]
struct TransferDiagnostics {
    domain_shift: f32,
    transferability_scores: Vec<f32>,
    recommended_strategy: TransferStrategy,
}

#[derive(Debug)]
enum TransferStrategy {
    FeatureExtraction,
    PartialFineTuning,
    FullFineTuning,
    DomainAdaptation,
    FromScratch,
}
```

## Conclusion

This tutorial covers the essential techniques for fine-tuning pre-trained models with scirs2-neural. The key takeaways are:

1. **Start with feature extraction** before full fine-tuning
2. **Use lower learning rates** for pre-trained layers
3. **Apply gradual unfreezing** for complex adaptations
4. **Monitor catastrophic forgetting** throughout training
5. **Use appropriate evaluation metrics** for your specific task
6. **Apply domain-specific augmentation** carefully

For more advanced techniques, consider exploring:
- Progressive neural networks
- Adapter layers
- LoRA (Low-Rank Adaptation)
- Knowledge distillation
- Multi-task learning approaches

Remember that fine-tuning is often task-specific, and the best approach depends on your data, computational resources, and performance requirements.

## Helper Functions

```rust
// Implementation of helper functions used in the examples above

fn argmax(array: &ArrayD<f32>) -> usize {
    array.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn evaluate_model(
    model: &Sequential<f32>,
    data: &[(ArrayD<f32>, usize)],
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut correct = 0;
    let mut total = 0;
    
    for (input, target) in data {
        let output = model.forward(input)?;
        let predicted = argmax(&output);
        
        if predicted == *target {
            correct += 1;
        }
        total += 1;
    }
    
    Ok((correct as f32 / total as f32) * 100.0)
}

fn train_epoch(
    model: &mut Sequential<f32>,
    data: &[(ArrayD<f32>, usize)],
    learning_rate: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut total_loss = 0.0;
    
    for (input, target) in data {
        let output = model.forward(input)?;
        
        // Simplified loss computation and backpropagation
        let loss = compute_loss(&output, *target)?;
        total_loss += loss;
        
        // Update parameters (simplified)
        model.update(learning_rate)?;
    }
    
    Ok(total_loss / data.len() as f32)
}

fn compute_loss(output: &ArrayD<f32>, target: usize) -> Result<f32, Box<dyn std::error::Error>> {
    // Simplified cross-entropy loss
    let softmax_output = softmax(output)?;
    Ok(-softmax_output[target].ln())
}

fn softmax(input: &ArrayD<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    
    Ok(exp_values.into_iter().map(|x| x / sum).collect())
}

// Additional helper functions would be implemented here...
```