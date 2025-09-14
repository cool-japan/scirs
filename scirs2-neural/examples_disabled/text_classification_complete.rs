//! Complete Text Classification Example
//!
//! This example demonstrates how to build, train, and evaluate a text classifier
//! using the scirs2-neural library. It covers:
//! - Text preprocessing and tokenization
//! - Building embedding-based models
//! - Training RNN/LSTM models for sequence classification
//! - Attention mechanisms for better performance
//! - Evaluation with text-specific metrics

use ndarray::{s, Array2, ArrayD};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use scirs2_neural::data::Dataset;
use scirs2_neural::layers::{Dense, Dropout};
use scirs2_neural::losses::CrossEntropyLoss;
use scirs2_neural::optimizers::Adam;
use scirs2_neural::prelude::*;
use scirs2_neural::training::{Trainer, TrainingConfig, ValidationSettings};
use std::collections::HashMap;
// Type alias to avoid conflicts with scirs2-neural's Result
type StdResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Simple vocabulary for text processing
#[derive(Debug, Clone)]
struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: HashMap<usize, String>,
    vocab_size: usize,
}
impl Vocabulary {
    fn new() -> Self {
        let mut vocab = Self {
            word_to_id: HashMap::new(),
            id_to_word: HashMap::new(),
            vocab_size: 0,
        };
        // Add special tokens
        vocab.add_word("<PAD>"); // Padding token
        vocab.add_word("<UNK>"); // Unknown token
        vocab.add_word("<START>"); // Start of sequence
        vocab.add_word("<END>"); // End of sequence
        vocab
    }
    fn add_word(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            let id = self.vocab_size;
            self.word_to_id.insert(word.to_string(), id);
            self.id_to_word.insert(id, word.to_string());
            self.vocab_size += 1;
        }
    fn word_to_id(&self, word: &str) -> usize {
        self.word_to_id.get(word).copied().unwrap_or(1) // Default to <UNK>
    fn tokenizetext(&self, text: &str, maxlength: usize) -> Vec<usize> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens = vec![2]; // Start with <START> token
        for word in words.iter().take(max_length - 2) {
            // Reserve space for START and END
            tokens.push(self.word_to_id(word));
        tokens.push(3); // Add <END> token
        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0); // <PAD> token
        tokens
/// Text classification dataset
#[derive(Clone)]
struct TextDataset {
    texts: Vec<String>,
    labels: Vec<usize>,
    vocab: Vocabulary,
    max_length: usize,
    num_classes: usize,
impl TextDataset {
    fn new(_max_length: usize, numclasses: usize) -> Self {
        let mut dataset = Self {
            texts: Vec::new(),
            labels: Vec::new(),
            vocab: Vocabulary::new(),
            max_length,
            num_classes,
        // Build vocabulary with common words
        let common_words = vec![
            "good",
            "bad",
            "great",
            "terrible",
            "excellent",
            "awful",
            "amazing",
            "horrible",
            "love",
            "hate",
            "like",
            "dislike",
            "enjoy",
            "fun",
            "boring",
            "interesting",
            "fast",
            "slow",
            "easy",
            "hard",
            "simple",
            "complex",
            "clear",
            "confusing",
            "helpful",
            "useless",
            "important",
            "trivial",
            "big",
            "small",
            "long",
            "short",
            "nice",
            "mean",
            "kind",
            "rude",
            "smart",
            "stupid",
            "funny",
            "serious",
            "happy",
            "sad",
            "excited",
            "disappointed",
            "satisfied",
            "frustrated",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
            "these",
            "those",
            "very",
            "quite",
            "really",
            "so",
            "movie",
            "book",
            "product",
            "service",
            "experience",
            "quality",
            "price",
        ];
        for word in common_words {
            dataset.vocab.add_word(word);
        dataset
    fn add_sample(&mut self, text: &str, label: usize) {
        // Add words from text to vocabulary
        for word in text.split_whitespace() {
            self.vocab.add_word(word);
        self.texts.push(text.to_string());
        self.labels.push(label);
    fn create_synthetic_dataset(_num_samples: usize, num_classes: usize, maxlength: usize) -> Self {
        let mut dataset = Self::new(_max_length, num_classes);
        let mut rng = SmallRng::from_seed([42; 32]);
        // Define sentiment patterns for each class
        let positive_words = vec![
            "fantastic",
            "wonderful",
        let negative_words = vec![
            "disappointing",
        let neutral_words = vec!["okay", "average", "normal", "fine", "decent", "standard"];
        let patterns = vec![
            (&positive_words, "This is a great product and I love it"),
            (&negative_words, "This is terrible and I hate it completely"),
            (
                &neutral_words,
                "This is an okay product with average quality",
            ),
        for _ in 0..num_samples {
            let class = rng.gen_range(0..num_classes.min(patterns.len()));
            let (words..base_pattern) = &patterns[class];
            // Generate synthetic text with class-specific words
            let num_extra_words = rng.gen_range(1..5);
            let mut text_parts = vec![base_pattern.to_string()];
            for _ in 0..num_extra_words {
                let word = words[rng.gen_range(0..words.len())];
                text_parts.push(format!("really {}"..word));
            }
            let text = text_parts.join(" ");
            dataset.add_sample(&text, class);
    fn train_val_split(&self, valratio: f32) -> (Self, Self) {
        let total_samples = self.len();
        let val_size = (total_samples as f32 * val_ratio) as usize;
        let train_size = total_samples - val_size;
        let train_dataset = Self {
            texts: self.texts[0..train_size].to_vec(),
            labels: self.labels[0..train_size].to_vec(),
            vocab: self.vocab.clone(),
            max_length: self.max_length,
            num_classes: self.num_classes,
        let val_dataset = Self {
            texts: self.texts[train_size..].to_vec(),
            labels: self.labels[train_size..].to_vec(),
        (train_dataset, val_dataset)
impl Dataset<f32> for TextDataset {
    fn len(&self) -> usize {
        self.texts.len()
    fn get(&self, index: usize) -> scirs2_neural::error::Result<(ArrayD<f32>, ArrayD<f32>)> {
        if index >= self.len() {
            return Err(scirs2_neural::error::NeuralError::InvalidArgument(format!(
                "Index {} out of bounds for dataset of size {}",
                index,
                self.len()
            )));
        // Tokenize text and convert to arrays
        let mut tokens_array = Array2::zeros((1, self.max_length));
        let mut label_array = Array2::zeros((1, self.num_classes));
        let tokens = self
            .vocab
            .tokenizetext(&self.texts[index], self.max_length);
        for (j, &token) in tokens.iter().enumerate() {
            tokens_array[[0, j]] = token as f32;
        // One-hot encode label
        label_array[[0, self.labels[index]]] = 1.0;
        Ok((tokens_array.into_dyn(), label_array.into_dyn()))
    fn box_clone(&self) -> Box<dyn Dataset<f32> + Send + Sync> {
        Box::new(self.clone())
/// Build a text classification model using embeddings and LSTM
#[allow(dead_code)]
fn buildtext_model(
    _vocab_size: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    rng: &mut SmallRng,
) -> StdResult<Sequential<f32>> {
    let mut model = Sequential::new();
    // Note: This is a simplified version since our Sequential model expects specific layer types
    // In a real implementation, we'd need specialized text layers
    // Flatten the input tokens and use dense layers to simulate embedding and LSTM processing
    let input_size = max_length; // Each token position
    // Simulate embedding layer with dense transformation
    model.add(Dense::new(
        input_size,
        embedding_dim * 2,
        Some("relu"),
        rng,
    )?);
    model.add(Dropout::new(0.1, rng)?);
    // Simulate LSTM processing with dense layers
        hidden_dim,
        Some("tanh"),
    model.add(Dropout::new(0.2, rng)?);
    // Add attention-like mechanism with another dense layer
    model.add(Dense::new(hidden_dim, hidden_dim / 2, Some("relu"), rng)?);
    // Classification head
        hidden_dim / 2,
        num_classes,
        Some("softmax"),
    Ok(model)
/// Build an advanced text model with attention
#[allow(dead_code)]
fn build_attentiontext_model(
    // Simplified attention-based model using dense layers
    let input_size = max_length;
    // Multi-layer processing to simulate transformer-like behavior
    model.add(Dense::new(input_size, embedding_dim, Some("relu"), rng)?);
    // Simulate self-attention with dense layers
    model.add(Dense::new(embedding_dim, hidden_dim, Some("relu"), rng)?);
    // Second attention layer
    model.add(Dense::new(hidden_dim, hidden_dim, Some("relu"), rng)?);
    // Global pooling simulation
/// Calculate text classification metrics
#[allow(dead_code)]
fn calculatetext_metrics(
    predictions: &ArrayD<f32>,
    targets: &ArrayD<f32>,
) -> HashMap<String, f32> {
    let batch_size = predictions.shape()[0];
    let num_classes = predictions.shape()[1];
    let mut metrics = HashMap::new();
    let mut correct = 0;
    let mut class_correct = vec![0; num_classes];
    let mut class_total = vec![0; num_classes];
    for i in 0..batch_size {
        let pred_row = predictions.slice(s![i, ..]);
        let target_row = targets.slice(s![i, ..]);
        // Find predicted and true classes
        let pred_class = pred_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i_)| i)
            .unwrap_or(0);
        let true_class = target_row
        class_total[true_class] += 1;
        if pred_class == true_class {
            correct += 1;
            class_correct[true_class] += 1;
    // Overall accuracy
    metrics.insert("accuracy".to_string(), correct as f32 / batch_size as f32);
    // Per-class accuracy
    for i in 0..num_classes {
        if class_total[i] > 0 {
            let class_acc = class_correct[i] as f32 / class_total[i] as f32;
            metrics.insert(format!("class_{}_accuracy", i), class_acc);
    metrics
/// Train text classification model
#[allow(dead_code)]
fn traintext_classifier() -> StdResult<()> {
    println!("📝 Starting Text Classification Training Example");
    println!("{}", "=".repeat(60));
    let mut rng = SmallRng::from_seed([42; 32]);
    // Dataset parameters
    let num_samples = 800;
    let num_classes = 3;
    let max_length = 20;
    let embedding_dim = 64;
    let hidden_dim = 128;
    println!("📊 Dataset Configuration:");
    println!("   - Samples: {}", num_samples);
    println!(
        "   - Classes: {} (Positive, Negative, Neutral)",
        num_classes
    );
    println!("   - Max sequence length: {}", max_length);
    println!("   - Embedding dimension: {}", embedding_dim);
    // Create synthetic text dataset
    println!("\n🔄 Creating synthetic text dataset...");
    let dataset = TextDataset::create_synthetic_dataset(num_samples, num_classes, max_length);
    let (train_dataset, val_dataset) = dataset.train_val_split(0.2);
    println!("   - Vocabulary size: {}", dataset.vocab.vocab_size);
    println!("   - Training samples: {}", train_dataset.len());
    println!("   - Validation samples: {}", val_dataset.len());
    // Show some example texts
    println!("\n📄 Sample texts:");
    for i in 0..3.min(train_dataset.texts.len()) {
        println!(
            "   [Class {}]: {}",
            train_dataset.labels[i], train_dataset.texts[i]
        );
    // Build model
    println!("\n🏗️  Building text classification model...");
    let model = buildtext_model(
        dataset.vocab.vocab_size,
        embedding_dim,
        max_length,
        &mut rng,
    )?;
    let total_params: usize = model.params().iter().map(|p| p.len()).sum();
    println!("   - Model layers: {}", model.len());
    println!("   - Total parameters: {}", total_params);
    // Training configuration
    let config = TrainingConfig {
        batch_size: 16,
        epochs: 30,
        learning_rate: 0.001,
        shuffle: true,
        verbose: 1,
        validation: Some(ValidationSettings {
            enabled: true,
            validation_split: 0.2,
            batch_size: 32,
            num_workers: 0,
        }),
        gradient_accumulation: None,
        mixed_precision: None,
        num_workers: 0,
    };
    println!("\n⚙️  Training Configuration:");
    println!("   - Batch size: {}", config.batch_size);
    println!("   - Learning rate: {}", config.learning_rate);
    println!("   - Epochs: {}", config.epochs);
    // Set up training
    let loss_fn = CrossEntropyLoss::new(1e-7);
    let optimizer = Adam::new(config.learning_rate as f32, 0.9, 0.999, 1e-8);
    let mut trainer = Trainer::new(model, optimizer, loss_fn, config);
    // Train the model
    println!("\n🏋️  Starting training...");
    println!("{}", "-".repeat(40));
    let training_session = trainer.train(&train_dataset, Some(&val_dataset))?;
    println!("\n✅ Training completed!");
    println!("   - Epochs trained: {}", training_session.epochs_trained);
    // Evaluate model
    println!("\n📊 Final Evaluation:");
    let val_metrics = trainer.validate(&val_dataset)?;
    for (metric, value) in &val_metrics {
        println!("   - {}: {:.4}", metric, value);
    // Test on sample texts
    println!("\n🔍 Sample Predictions:");
    let sample_indices = vec![0, 1, 2, 3, 4];
    // Manually collect batch since get_batch is not part of Dataset trait
    let mut batch_tokens = Vec::new();
    let mut batch_targets = Vec::new();
    for &idx in &sample_indices {
        let (tokens, targets) = val_dataset.get(idx)?;
        batch_tokens.push(tokens);
        batch_targets.push(targets);
    // Concatenate into batch arrays
    let sample_tokens = ndarray::concatenate(
        ndarray::Axis(0),
        &batch_tokens.iter().map(|a| a.view()).collect::<Vec<_>>(),
    let sample_targets = ndarray::concatenate(
        &batch_targets.iter().map(|a| a.view()).collect::<Vec<_>>(),
    let model = trainer.get_model();
    let predictions = model.forward(&sample_tokens)?;
    let class_names = ["Positive", "Negative", "Neutral"];
    for i in 0..sample_indices.len().min(val_dataset.texts.len()) {
        let target_row = sample_targets.slice(s![i, ..]);
        let confidence = pred_row[pred_class];
        if sample_indices[i] < val_dataset.texts.len() {
            println!("   Text: \"{}\"", val_dataset.texts[sample_indices[i]]);
            println!(
                "   Predicted: {} (confidence: {:.3})",
                class_names[pred_class], confidence
            );
            println!("   Actual: {}", class_names[true_class]);
            println!();
    // Calculate detailed metrics
    let detailed_metrics = calculatetext_metrics(&predictions, &sample_targets);
    println!("📈 Detailed Metrics:");
    for (metric, value) in &detailed_metrics {
    Ok(())
/// Demonstrate advanced text model with attention
#[allow(dead_code)]
fn demonstrate_attention_model() -> StdResult<()> {
    println!("\n🎯 Attention-Based Model Demo:");
    let mut rng = SmallRng::from_seed(123);
    // Create attention model
    let model = build_attentiontext_model(1000, 128, 256, 3, 20, &mut rng)?;
    println!model.params(.iter().map(|p| p.len()).sum::<usize>()
    println!("   ✅ Attention mechanism simulation completed");
/// Demonstrate text preprocessing
#[allow(dead_code)]
fn demonstratetext_preprocessing() -> StdResult<()> {
    println!("\n🔤 Text Preprocessing Demo:");
    println!("{}", "-".repeat(30));
    let mut vocab = Vocabulary::new();
    // Add some sample words
    let sampletext = "This is a great movie and I really love it";
    for word in sampletext.split_whitespace() {
        vocab.add_word(word);
    // Tokenize text
    let tokens = vocab.tokenizetext(sampletext, 15);
    println!("   - Original text: \"{}\"", sampletext);
    println!("   - Vocabulary size: {}", vocab.vocab_size);
    println!("   - Tokenized: {:?}", tokens);
    // Show token to word mapping for first few tokens
    println!("   - Token meanings:");
    for (_i, &token_id) in tokens.iter().take(8).enumerate() {
        if let Some(word) = vocab.id_to_word.get(&token_id) {
            println!("     [{}] -> \"{}\"", token_id, word);
    println!("   ✅ Text preprocessing completed");
/// Main function
#[allow(dead_code)]
fn main() -> StdResult<()> {
    // Main training example
    traintext_classifier()?;
    // Additional demonstrations
    demonstrate_attention_model()?;
    demonstratetext_preprocessing()?;
    println!("\n🌟 All text classification examples completed successfully!");
    println!("🔗 Next steps:");
    println!("   - Try with real text datasets (IMDB, Reuters, etc.)");
    println!("   - Implement proper embedding layers");
    println!("   - Add BERT-like pre-trained models");
    println!("   - Experiment with different tokenization strategies");
    println!("   - Add more sophisticated attention mechanisms");
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vocabulary_creation() {
        let mut vocab = Vocabulary::new();
        let word_id = vocab.add_word("test");
        assert_eq!(vocab.word_to_id("test"), word_id);
        assert_eq!(vocab.vocab_size, 5); // 4 special tokens + 1 added word
    fn testtext_tokenization() {
        vocab.add_word("hello");
        vocab.add_word("world");
        let tokens = vocab.tokenizetext("hello world", 5);
        assert_eq!(tokens.len(), 5); // Should be padded to length 5
        assert_eq!(tokens[0], 2); // <START> token
        assert_eq!(tokens[3], 3); // <END> token
    fn test_dataset_creation() {
        let dataset = TextDataset::create_synthetic_dataset(10, 3, 15);
        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.num_classes, 3);
        assert_eq!(dataset.max_length, 15);
    fn test_model_creation() -> StdResult<()> {
        let model = buildtext_model(100, 32, 64, 3, 10, &mut rng)?;
        assert!(!model.is_empty());
        Ok(())
    fn test_metrics_calculation() {
        let predictions = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.1, 0.8, 0.1, // Class 1
                0.7, 0.2, 0.1, // Class 0
            ],
        )
        .unwrap()
        .into_dyn();
        let targets = Array2::from_shape_vec(
                0.0, 1.0, 0.0, // Class 1
                1.0, 0.0, 0.0, // Class 0
        let metrics = calculatetext_metrics(&predictions, &targets);
        assert_eq!(metrics["accuracy"], 1.0);
