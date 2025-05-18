//! Text classification example

use scirs2_text::{
    TextClassificationMetrics, TextClassificationPipeline, TextDataset, TextFeatureSelector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Text Classification Demo");
    println!("=======================\n");

    // Create sample dataset
    let texts = vec![
        "This movie is absolutely fantastic and amazing!".to_string(),
        "I really hated this film, it was terrible.".to_string(),
        "The acting was superb and the plot was engaging.".to_string(),
        "Worst movie I've ever seen, complete waste of time.".to_string(),
        "A masterpiece of cinema, truly exceptional work.".to_string(),
        "Boring, predictable, and poorly executed.".to_string(),
    ];

    let labels = vec![
        "positive".to_string(),
        "negative".to_string(),
        "positive".to_string(),
        "negative".to_string(),
        "positive".to_string(),
        "negative".to_string(),
    ];

    // Create dataset
    let dataset = TextDataset::new(texts, labels)?;
    println!("Dataset Statistics:");
    println!("  Total samples: {}", dataset.len());
    println!("  Number of classes: {}", dataset.unique_labels().len());
    println!();

    // Split into train and test
    let (train_dataset, test_dataset) = dataset.train_test_split(0.33, Some(42))?;
    println!("Train/Test Split:");
    println!("  Training samples: {}", train_dataset.len());
    println!("  Test samples: {}", test_dataset.len());
    println!();

    // Create text processing pipeline
    let mut pipeline = TextClassificationPipeline::with_tfidf();

    // Fit the pipeline
    pipeline.fit(&train_dataset)?;

    // Transform to features
    let train_features = pipeline.transform(&train_dataset)?;
    let test_features = pipeline.transform(&test_dataset)?;

    println!("Feature Extraction:");
    println!(
        "  Train feature shape: ({}, {})",
        train_features.nrows(),
        train_features.ncols()
    );
    println!(
        "  Test feature shape: ({}, {})",
        test_features.nrows(),
        test_features.ncols()
    );
    println!();

    // Demonstrate feature selection
    let mut feature_selector = TextFeatureSelector::new()
        .set_max_features(10.0)?
        .set_min_df(0.1)?
        .set_max_df(0.9)?;

    let selected_train_features = feature_selector.fit_transform(&train_features)?;
    println!("Feature Selection:");
    println!("  Selected features: {}", selected_train_features.ncols());
    println!();

    // Simulate classification results (in a real scenario, you'd use a classifier)
    // For demo purposes, we'll create mock predictions based on simple heuristics
    let _unique_labels = train_dataset.unique_labels();

    // Create binary labels (0 for negative, 1 for positive) for this demo
    let mut _train_labels = Vec::new();
    let mut test_labels = Vec::new();

    for label in &train_dataset.labels {
        _train_labels.push(if label == "positive" { 1 } else { 0 });
    }

    for label in &test_dataset.labels {
        test_labels.push(if label == "positive" { 1 } else { 0 });
    }

    // Mock predictions (in practice, use a real classifier)
    let predictions = test_labels.clone(); // Perfect predictions for demo

    // Calculate metrics
    let metrics = TextClassificationMetrics::new();
    let accuracy = metrics.accuracy(&predictions, &test_labels)?;
    let (precision, recall, f1) = metrics.binary_metrics(&predictions, &test_labels)?;

    println!("Classification Metrics:");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Precision: {:.2}%", precision * 100.0);
    println!("  Recall: {:.2}%", recall * 100.0);
    println!("  F1 Score: {:.2}%", f1 * 100.0);
    println!();

    // Create a simple confusion matrix manually since the method isn't available
    let mut true_positive = 0;
    let mut true_negative = 0;
    let mut false_positive = 0;
    let mut false_negative = 0;

    for (pred, actual) in predictions.iter().zip(test_labels.iter()) {
        match (pred, actual) {
            (1, 1) => true_positive += 1,
            (0, 0) => true_negative += 1,
            (1, 0) => false_positive += 1,
            (0, 1) => false_negative += 1,
            _ => {}
        }
    }

    println!("Confusion Matrix:");
    println!("[ {} {} ]", true_negative, false_positive);
    println!("[ {} {} ]", false_negative, true_positive);

    Ok(())
}
