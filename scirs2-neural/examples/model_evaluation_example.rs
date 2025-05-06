use ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::error::Result;
use scirs2_neural::{
    activations::ReLU,
    data::{DataLoader, Dataset, InMemoryDataset},
    evaluation::{
        CrossValidationConfig, CrossValidationStrategy, CrossValidator, EarlyStoppingConfig,
        EarlyStoppingMode, EvaluationConfig, Evaluator, MetricType, TestConfig, TestEvaluator,
        ValidationConfig, ValidationHandler,
    },
    layers::{Dense, Dropout, Layer, Sequential},
    losses::{Loss, MSELoss},
    models::ModelBuilder,
};

use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;

// Simple model builder for testing
struct SimpleModelBuilder<F: Float + Debug + ScalarOperand> {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + Debug + ScalarOperand> SimpleModelBuilder<F> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: Float + Debug + ScalarOperand> ModelBuilder<F> for SimpleModelBuilder<F> {
    type Model = Sequential<F>;

    fn build(&self) -> Result<Self::Model> {
        let mut model = Sequential::new();

        model.add(Dense::<F>::new(
            self.input_dim,
            self.hidden_dim,
            Some(Box::new(ReLU::new())),
            None,
            Some("dense1"),
        )?);

        model.add(Dropout::<F>::new(0.2, Some("dropout1"))?);

        model.add(Dense::<F>::new(
            self.hidden_dim,
            self.output_dim,
            None,
            None,
            Some("dense2"),
        )?);

        Ok(model)
    }
}

// Generate synthetic regression dataset
fn generate_regression_dataset<F: Float + Debug + ScalarOperand>(
    n_samples: usize,
    input_dim: usize,
) -> InMemoryDataset<F> {
    let mut rng = rand::rng();

    // Generate random inputs
    let mut inputs = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut input = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            input.push(F::from(rng.gen_range(0.0..1.0)).unwrap());
        }
        inputs.push(
            Array::<F, _>::from_shape_vec([1, input_dim], input)
                .unwrap()
                .into_dyn(),
        );
    }

    // Generate targets (simple linear relationship plus noise)
    let mut targets = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let input = &inputs[i];
        let mut target_val = F::zero();

        for j in 0..input_dim {
            target_val = target_val + input[[0, j]] * F::from(j as f64 / input_dim as f64).unwrap();
        }

        // Add noise
        let noise = F::from(rng.gen_range(-0.1..0.1)).unwrap();
        target_val = target_val + noise;

        targets.push(
            Array::<F, _>::from_shape_vec([1, 1], vec![target_val])
                .unwrap()
                .into_dyn(),
        );
    }

    InMemoryDataset::new(inputs, targets)
}

// Generate synthetic classification dataset
fn generate_classification_dataset<F: Float + Debug + ScalarOperand>(
    n_samples: usize,
    input_dim: usize,
    n_classes: usize,
) -> InMemoryDataset<F> {
    let mut rng = rand::rng();

    // Generate random inputs
    let mut inputs = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut input = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            input.push(F::from(rng.gen_range(0.0..1.0)).unwrap());
        }
        inputs.push(
            Array::<F, _>::from_shape_vec([1, input_dim], input)
                .unwrap()
                .into_dyn(),
        );
    }

    // Generate targets (one-hot encoded)
    let mut targets = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let input = &inputs[i];
        let mut class_scores = Vec::with_capacity(n_classes);

        for c in 0..n_classes {
            let mut score = F::zero();
            for j in 0..input_dim {
                let weight = F::from(((c + j) % input_dim) as f64 / input_dim as f64).unwrap();
                score = score + input[[0, j]] * weight;
            }
            class_scores.push(score);
        }

        // Find max score
        let mut max_class = 0;
        let mut max_score = class_scores[0];
        for c in 1..n_classes {
            if class_scores[c] > max_score {
                max_score = class_scores[c];
                max_class = c;
            }
        }

        // Create one-hot target
        let mut one_hot = vec![F::zero(); n_classes];
        one_hot[max_class] = F::one();

        targets.push(
            Array::<F, _>::from_shape_vec([1, n_classes], one_hot)
                .unwrap()
                .into_dyn(),
        );
    }

    InMemoryDataset::new(inputs, targets)
}

fn main() -> Result<()> {
    println!("Model Evaluation Framework Example");
    println!("---------------------------------");

    // 1. Basic evaluation
    println!("\n1. Basic Evaluation:");

    // Generate synthetic regression dataset
    let dataset = generate_regression_dataset::<f32>(1000, 5);

    // Split into train, validation, and test sets
    let n_samples = dataset.len();
    let train_size = n_samples * 6 / 10;
    let val_size = n_samples * 2 / 10;
    let test_size = n_samples - train_size - val_size;

    let mut indices: Vec<usize> = (0..n_samples).collect();
    use rand::seq::SliceRandom;
    indices.shuffle(&mut rand::rng());

    let train_indices = indices[0..train_size].to_vec();
    let val_indices = indices[train_size..train_size + val_size].to_vec();
    let test_indices = indices[train_size + val_size..].to_vec();

    println!(
        "Dataset splits: Train={}, Validation={}, Test={}",
        train_indices.len(),
        val_indices.len(),
        test_indices.len()
    );

    // Create dataset views
    let train_dataset = scirs2_neural::data::DatasetView::new(&dataset, &train_indices);
    let val_dataset = scirs2_neural::data::DatasetView::new(&dataset, &val_indices);
    let test_dataset = scirs2_neural::data::DatasetView::new(&dataset, &test_indices);

    // Build a simple model
    let model_builder = SimpleModelBuilder::<f32>::new(5, 32, 1);
    let mut model = model_builder.build()?;

    // Create loss function
    let loss_fn = MSELoss::new();

    // Create evaluator
    let eval_config = EvaluationConfig {
        batch_size: 32,
        shuffle: false,
        num_workers: 0,
        metrics: vec![
            MetricType::Loss,
            MetricType::MeanSquaredError,
            MetricType::MeanAbsoluteError,
            MetricType::RSquared,
        ],
        steps: None,
        verbose: 1,
    };

    let mut evaluator = Evaluator::new(eval_config)?;

    // Evaluate model on validation set
    println!("\nEvaluating model on validation set:");
    let val_metrics = evaluator.evaluate(&model, &val_dataset, Some(&loss_fn))?;

    println!("Validation metrics:");
    for (name, value) in &val_metrics {
        println!("  {}: {:.4}", name, value);
    }

    // 2. Validation with early stopping
    println!("\n2. Validation with Early Stopping:");

    // Configure early stopping
    let early_stopping_config = EarlyStoppingConfig {
        monitor: "val_loss".to_string(),
        min_delta: 0.001,
        patience: 5,
        restore_best_weights: true,
        mode: EarlyStoppingMode::Min,
    };

    let validation_config = ValidationConfig {
        batch_size: 32,
        shuffle: false,
        num_workers: 0,
        steps: None,
        metrics: vec![MetricType::Loss, MetricType::MeanSquaredError],
        verbose: 1,
        early_stopping: Some(early_stopping_config),
    };

    let mut validation_handler = ValidationHandler::new(validation_config)?;

    // Simulate training loop with validation
    println!("\nSimulating training loop with validation:");
    let num_epochs = 10;

    for epoch in 0..num_epochs {
        println!("Epoch {}/{}", epoch + 1, num_epochs);

        // Simulate training step (not actually training the model)
        println!("Training...");

        // Validate model
        let (val_metrics, should_stop) =
            validation_handler.validate(&mut model, &val_dataset, Some(&loss_fn), epoch)?;

        println!("Validation metrics:");
        for (name, value) in &val_metrics {
            println!("  {}: {:.4}", name, value);
        }

        if should_stop {
            println!("Early stopping triggered!");
            break;
        }
    }

    // 3. Cross-validation
    println!("\n3. Cross-Validation:");

    // Configure cross-validation
    let cv_config = CrossValidationConfig {
        strategy: CrossValidationStrategy::KFold(5),
        shuffle: true,
        random_seed: Some(42),
        batch_size: 32,
        num_workers: 0,
        metrics: vec![
            MetricType::Loss,
            MetricType::MeanSquaredError,
            MetricType::RSquared,
        ],
        verbose: 1,
    };

    let mut cross_validator = CrossValidator::new(cv_config)?;

    // Perform cross-validation
    println!("\nPerforming 5-fold cross-validation:");
    let cv_results = cross_validator.cross_validate(&model_builder, &dataset, Some(&loss_fn))?;

    println!("Cross-validation results:");
    for (name, values) in &cv_results {
        // Calculate mean and std
        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;

        let variance_sum: f32 = values.iter().map(|&x| (x - mean).powi(2)).sum();
        let std = (variance_sum / values.len() as f32).sqrt();

        println!("  {}: {:.4} ± {:.4}", name, mean, std);
    }

    // 4. Test set evaluation
    println!("\n4. Test Set Evaluation:");

    // Configure test evaluator
    let test_config = TestConfig {
        batch_size: 32,
        num_workers: 0,
        metrics: vec![
            MetricType::Loss,
            MetricType::MeanSquaredError,
            MetricType::MeanAbsoluteError,
            MetricType::RSquared,
        ],
        steps: None,
        verbose: 1,
        generate_predictions: true,
        save_outputs: false,
    };

    let mut test_evaluator = TestEvaluator::new(test_config)?;

    // Evaluate model on test set
    println!("\nEvaluating model on test set:");
    let test_metrics = test_evaluator.evaluate(&model, &test_dataset, Some(&loss_fn))?;

    println!("Test metrics:");
    for (name, value) in &test_metrics {
        println!("  {}: {:.4}", name, value);
    }

    // 5. Classification example
    println!("\n5. Classification Example:");

    // Generate synthetic classification dataset
    let n_classes = 3;
    let class_dataset = generate_classification_dataset::<f32>(1000, 5, n_classes);

    // Split dataset
    let class_train_dataset = scirs2_neural::data::DatasetView::new(&class_dataset, &train_indices);
    let class_test_dataset = scirs2_neural::data::DatasetView::new(&class_dataset, &test_indices);

    // Build classification model
    let class_model_builder = SimpleModelBuilder::<f32>::new(5, 32, n_classes);
    let class_model = class_model_builder.build()?;

    // Configure test evaluator for classification
    let class_test_config = TestConfig {
        batch_size: 32,
        num_workers: 0,
        metrics: vec![
            MetricType::Accuracy,
            MetricType::Precision,
            MetricType::Recall,
            MetricType::F1Score,
        ],
        steps: None,
        verbose: 1,
        generate_predictions: true,
        save_outputs: false,
    };

    let mut class_test_evaluator = TestEvaluator::new(class_test_config)?;

    // Evaluate classification model
    println!("\nEvaluating classification model:");
    let class_metrics = class_test_evaluator.evaluate(&class_model, &class_test_dataset, None)?;

    println!("Classification metrics:");
    for (name, value) in &class_metrics {
        println!("  {}: {:.4}", name, value);
    }

    // Generate classification report
    println!("\nClassification Report:");
    match class_test_evaluator.classification_report() {
        Ok(report) => println!("{}", report),
        Err(e) => println!("Could not generate classification report: {}", e),
    }

    // Generate confusion matrix
    println!("\nConfusion Matrix:");
    match class_test_evaluator.confusion_matrix() {
        Ok(cm) => println!("{}", cm),
        Err(e) => println!("Could not generate confusion matrix: {}", e),
    }

    println!("\nModel Evaluation Example Completed Successfully!");

    Ok(())
}
