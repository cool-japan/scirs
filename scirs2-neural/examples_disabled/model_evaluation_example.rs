use ndarray::{Array, ScalarOperand};
use scirs2_neural::error::Result;
use scirs2_neural::{
    data::{Dataset, InMemoryDataset, SubsetDataset},
    evaluation::{
        CrossValidationConfig, CrossValidationStrategy, CrossValidator, EarlyStoppingConfig,
        EarlyStoppingMode, EvaluationConfig, Evaluator, MetricType, ModelBuilder, TestConfig,
        TestEvaluator, ValidationConfig, ValidationHandler,
    },
    layers::{Dense, Dropout, Sequential},
    losses::MeanSquaredError,
};

use num_traits::{Float, FromPrimitive};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::fmt::Debug;
use std::marker::{Send, Sync};
// Simple model builder for testing
struct SimpleModelBuilder<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize, _phantom: std::marker::PhantomData<F>,
}
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> SimpleModelBuilder<F> {
    fn new(_input_dim: usize, hidden_dim: usize, outputdim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim_phantom: std::marker::PhantomData,
        }
    }
impl<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync> ModelBuilder<F>
    for SimpleModelBuilder<F>
{
    type Model = Sequential<F>;
    fn build(&self) -> Result<Self::Model> {
        let mut model = Sequential::new();
        let mut rng = SmallRng::from_seed([42; 32]);
        model.add(Dense::<F>::new(
            self.input_dim,
            self.hidden_dim,
            Some("relu"),
            &mut rng,
        )?);
        model.add(Dropout::<F>::new(0.2, &mut rng)?);
            self.output_dim,
            None,
        Ok(model)
// Generate synthetic regression dataset
#[allow(dead_code)]
fn generate_regression_dataset<F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync>(
    n_samples: usize,
) -> Result<InMemoryDataset<F>> {
    let mut rng = SmallRng::from_seed([42; 32]);
    // Generate random inputs directly into a single array
    let mut features_data = Vec::with_capacity(n_samples * input_dim);
    for _ in 0..n_samples {
        for _ in 0..input_dim {
            features_data.push(F::from(rng.gen_range(0.0..1.0)).unwrap());
    let features = Array::<F.._>::from_shape_vec([n_samples, input_dim], features_data)
        .unwrap()
        .into_dyn();
    // Generate targets (simple linear relationship plus noise)
    let mut labels_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut target_val = F::zero();
        for j in 0..input_dim {
            target_val =
                target_val + features[[i, j]] * F::from(j as f64 / input_dim as f64).unwrap();
        // Add noise
        let noise = F::from(rng.gen_range(-0.1..0.1)).unwrap();
        target_val = target_val + noise;
        labels_data.push(target_val);
    let labels = Array::<F.._>::from_shape_vec([n_samples, 1], labels_data)
    InMemoryDataset::new(features, labels)
// Generate synthetic classification dataset
#[allow(dead_code)]
fn generate_classification_dataset<
    F: Float + Debug + ScalarOperand + FromPrimitive + Send + Sync,
>(
    n_classes: usize,
    let mut rng = SmallRng::from_seed(43);
    // Generate targets (one-hot encoded)
    let mut labels_data = vec![F::zero(); n_samples * n_classes];
        let mut class_scores = Vec::with_capacity(n_classes);
        for c in 0..n_classes {
            let mut score = F::zero();
            for j in 0..input_dim {
                let weight = F::from(((c + j) % input_dim) as f64 / input_dim as f64).unwrap();
                score = score + features[[i, j]] * weight;
            }
            class_scores.push(score);
        // Find max score
        let mut max_class = 0;
        let mut max_score = class_scores[0];
        for c in 1..n_classes {
            if class_scores[c] > max_score {
                max_score = class_scores[c];
                max_class = c;
        // Set the one-hot encoding in the labels array
        labels_data[i * n_classes + max_class] = F::one();
    let labels = Array::<F>::from_shape_vec([n_samples, n_classes], labels_data)
#[allow(dead_code)]
fn main() -> Result<()> {
    println!("Model Evaluation Framework Example");
    println!("---------------------------------");
    // 1. Basic evaluation
    println!("\n1. Basic Evaluation:");
    // Generate synthetic regression dataset
    let dataset = generate_regression_dataset::<f32>(1000, 5)?;
    // Split into train, validation, and test sets
    let n_samples = dataset.len();
    let train_size = n_samples * 6 / 10;
    let val_size = n_samples * 2 / 10;
    let _test_size = n_samples - train_size - val_size;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    use rand::seq::SliceRandom;
    let mut shuffle_rng = SmallRng::from_seed(44);
    indices.shuffle(&mut shuffle_rng);
    let train_indices = indices[0..train_size].to_vec();
    let val_indices = indices[train_size..train_size + val_size].to_vec();
    let test_indices = indices[train_size + val_size..].to_vec();
    println!(
        "Dataset splits: Train={}, Validation={}, Test={}",
        train_indices.len(),
        val_indices.len(),
        test_indices.len()
    );
    // Create subset datasets
    let _train_dataset = SubsetDataset::new(dataset.clone(), train_indices.clone())?;
    let val_dataset = SubsetDataset::new(dataset.clone(), val_indices.clone())?;
    let test_dataset = SubsetDataset::new(dataset.clone(), test_indices.clone())?;
    // Build a simple model
    let model_builder = SimpleModelBuilder::<f32>::new(5, 32, 1);
    let mut model = model_builder.build()?;
    // Create loss function
    let loss_fn = MeanSquaredError::new();
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
    let val_metrics = evaluator.evaluate(&mut model, &val_dataset, Some(&loss_fn))?;
    println!("Validation metrics:");
    for (name, value) in &val_metrics {
        println!("  {}: {:.4}", name, value);
    // 2. Validation with early stopping
    println!("\n2. Validation with Early Stopping:");
    // Configure early stopping
    let early_stopping_config = EarlyStoppingConfig {
        monitor: "val_loss".to_string(),
        min_delta: 0.001,
        patience: 5,
        restore_best_weights: true,
        mode: EarlyStoppingMode::Min,
    let validation_config = ValidationConfig {
        metrics: vec![MetricType::Loss, MetricType::MeanSquaredError],
        early_stopping: Some(early_stopping_config),
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
        if should_stop {
            println!("Early stopping triggered!");
            break;
    // 3. Cross-validation
    println!("\n3. Cross-Validation:");
    // Configure cross-validation
    let cv_config = CrossValidationConfig {
        strategy: CrossValidationStrategy::KFold(5),
        shuffle: true,
        random_seed: Some(42),
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
    // 4. Test set evaluation
    println!("\n4. Test Set Evaluation:");
    // Configure test evaluator
    let test_config = TestConfig {
        generate_predictions: true,
        save_outputs: false,
    let mut test_evaluator = TestEvaluator::new(test_config)?;
    // Evaluate model on test set
    println!("\nEvaluating model on test set:");
    let test_metrics = test_evaluator.evaluate(&mut model, &test_dataset, Some(&loss_fn))?;
    println!("Test metrics:");
    for (name, value) in &test_metrics {
    // 5. Classification example
    println!("\n5. Classification Example:");
    // Generate synthetic classification dataset
    let n_classes = 3;
    let class_dataset = generate_classification_dataset::<f32>(1000, 5, n_classes)?;
    // Split dataset
    let _class_train_dataset = SubsetDataset::new(class_dataset.clone(), train_indices.clone())?;
    let class_test_dataset = SubsetDataset::new(class_dataset.clone(), test_indices.clone())?;
    // Build classification model
    let class_model_builder = SimpleModelBuilder::<f32>::new(5, 32, n_classes);
    let mut class_model = class_model_builder.build()?;
    // Configure test evaluator for classification
    let class_test_config = TestConfig {
            MetricType::Accuracy,
            MetricType::Precision,
            MetricType::Recall,
            MetricType::F1Score,
    let mut class_test_evaluator = TestEvaluator::new(class_test_config)?;
    // Evaluate classification model
    println!("\nEvaluating classification model:");
    let class_metrics =
        class_test_evaluator.evaluate(&mut class_model, &class_test_dataset, None)?;
    println!("Classification metrics:");
    for (name, value) in &class_metrics {
    // Generate classification report
    println!("\nClassification Report:");
    match class_test_evaluator.classification_report() {
        Ok(report) => println!("{}", report),
        Err(e) => println!("Could not generate classification report: {}", e),
    // Generate confusion matrix
    println!("\nConfusion Matrix:");
    match class_test_evaluator.confusion_matrix() {
        Ok(cm) => println!("{}", cm),
        Err(e) => println!("Could not generate confusion matrix: {}", e),
    println!("\nModel Evaluation Example Completed Successfully!");
    Ok(())
