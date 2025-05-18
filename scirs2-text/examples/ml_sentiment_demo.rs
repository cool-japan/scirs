//! Machine learning based sentiment analysis demonstration

use scirs2_text::{MLSentimentAnalyzer, MLSentimentConfig, TextDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ML-based Sentiment Analysis Demo");
    println!("================================\n");

    // Create a sample dataset
    let (train_dataset, test_dataset) = create_sentiment_dataset()?;

    println!("1. Dataset Information");
    println!("--------------------");
    println!("Training examples: {}", train_dataset.texts.len());
    println!("Test examples: {}", test_dataset.texts.len());

    let labels_train: std::collections::HashSet<_> = train_dataset.labels.iter().cloned().collect();
    println!("Labels: {:?}\n", labels_train);

    // Configure and train ML sentiment analyzer
    println!("2. Training ML Sentiment Analyzer");
    println!("------------------------------");

    let config = MLSentimentConfig {
        learning_rate: 0.05,
        epochs: 200,
        regularization: 0.01,
        batch_size: 32,
        random_seed: Some(42),
    };

    let mut analyzer = MLSentimentAnalyzer::new().with_config(config);

    println!("Training...");
    let training_metrics = analyzer.train(&train_dataset)?;

    println!("Training complete!");
    println!("Final accuracy: {:.4}", training_metrics.accuracy);
    println!(
        "Final loss: {:.4}",
        training_metrics.loss_history.last().unwrap()
    );

    // Plot loss history
    println!("\nLoss history (first 10 epochs):");
    print!("  ");
    for i in 0..10 {
        print!("{:.2} ", training_metrics.loss_history[i]);
    }
    println!("...");

    // Evaluate on test data
    println!("\n3. Evaluation");
    println!("-----------");

    let eval_metrics = analyzer.evaluate(&test_dataset)?;

    println!("Accuracy: {:.4}", eval_metrics.accuracy);
    println!("Precision: {:.4}", eval_metrics.precision);
    println!("Recall: {:.4}", eval_metrics.recall);
    println!("F1 Score: {:.4}", eval_metrics.f1_score);

    println!("\nClass metrics:");
    for (label, metrics) in &eval_metrics.class_metrics {
        println!(
            "  {}: Precision={:.4}, Recall={:.4}, F1={:.4}",
            label, metrics.precision, metrics.recall, metrics.f1_score
        );
    }

    // Display confusion matrix
    println!("\nConfusion Matrix:");
    for i in 0..eval_metrics.confusion_matrix.nrows() {
        print!("  ");
        for j in 0..eval_metrics.confusion_matrix.ncols() {
            print!("{:4} ", eval_metrics.confusion_matrix[[i, j]]);
        }
        println!();
    }

    // Use the model for predictions
    println!("\n4. Sentiment Predictions");
    println!("----------------------");

    let test_texts = vec![
        "This product is amazing! I absolutely love it and would recommend it to everyone.",
        "Terrible experience. The customer service was awful and the product doesn't work.",
        "It's okay. Not great, not terrible, just average.",
        "Good value for money, but there are some issues with the packaging.",
        "Worst purchase ever. Complete waste of money.",
    ];

    println!("Sample text predictions:");
    for text in test_texts {
        let result = analyzer.predict(text)?;
        println!(
            "\"{}...\"\n  → {} (Score: {:.2}, Confidence: {:.2})\n",
            text.chars().take(40).collect::<String>(),
            result.sentiment,
            result.score,
            result.confidence
        );
    }

    // Batch prediction
    println!("5. Batch Prediction");
    println!("----------------");

    let batch_texts = vec![
        "Excellent quality product",
        "Poor performance for the price",
        "Somewhat satisfied with purchase",
    ];

    let batch_results = analyzer.predict_batch(&batch_texts)?;

    for (i, result) in batch_results.iter().enumerate() {
        println!(
            "Text {}: {} (Confidence: {:.2})",
            i + 1,
            result.sentiment,
            result.confidence
        );
    }

    // Compare with different configurations
    println!("\n6. Hyperparameter Comparison");
    println!("--------------------------");

    let configs = vec![
        (0.01, 100, "Low learning rate"),
        (0.1, 100, "High learning rate"),
        (0.05, 50, "Medium rate, fewer epochs"),
        (0.05, 200, "Medium rate, more epochs"),
    ];

    for (lr, epochs, desc) in configs {
        let config = MLSentimentConfig {
            learning_rate: lr,
            epochs,
            regularization: 0.01,
            batch_size: 32,
            random_seed: Some(42),
        };

        let mut temp_analyzer = MLSentimentAnalyzer::new().with_config(config);
        let _metrics = temp_analyzer.train(&train_dataset)?;
        let eval = temp_analyzer.evaluate(&test_dataset)?;

        println!(
            "{}: Accuracy={:.4}, F1={:.4}",
            desc, eval.accuracy, eval.f1_score
        );
    }

    Ok(())
}

fn create_sentiment_dataset() -> Result<(TextDataset, TextDataset), Box<dyn std::error::Error>> {
    // Training data
    let train_texts = vec![
        "I absolutely loved this movie! The acting was superb.",
        "Terrible experience, would not recommend to anyone.",
        "The product was okay, nothing special but it works.",
        "Great customer service and fast delivery.",
        "Disappointing quality for the price paid.",
        "This is the best purchase I've made all year!",
        "Waste of money, doesn't work as advertised.",
        "Mixed feelings about this. Some parts good, others bad.",
        "Pleasantly surprised by how well this performs.",
        "Not worth the price. Broke after two weeks.",
        "Amazing value for the price. Highly recommended!",
        "Mediocre at best. Wouldn't buy again.",
        "Fantastic product that exceeds expectations.",
        "Poor construction quality, arrived damaged.",
        "It's decent but there are better options available.",
        "This changed my life! Can't imagine living without it.",
        "Regret buying this. Customer service was unhelpful.",
        "Satisfied with my purchase, does what it claims.",
        "Best in its class. Outstanding performance.",
        "Very disappointed, doesn't match the description.",
        "Just average, nothing to write home about.",
        "Exceeded my expectations in every way.",
        "One of the worst products I've ever bought.",
        "Good enough for the price, but has limitations.",
        "Incredible value! Works perfectly for my needs.",
        "Would not purchase again. Many flaws.",
        "Does the job fine, but nothing spectacular.",
        "Absolutely worthless. Don't waste your money.",
        "A solid choice. Reliable and well-designed.",
        "Not impressed at all. Many issues from day one.",
    ];

    let train_labels = vec![
        "positive", "negative", "neutral", "positive", "negative", "positive", "negative",
        "neutral", "positive", "negative", "positive", "negative", "positive", "negative",
        "neutral", "positive", "negative", "neutral", "positive", "negative", "neutral",
        "positive", "negative", "neutral", "positive", "negative", "neutral", "negative",
        "positive", "negative",
    ];

    // Test data (different examples)
    let test_texts = vec![
        "Loved every minute of it. Highly recommended!",
        "Terrible product. Complete waste of money.",
        "It's okay, nothing special but gets the job done.",
        "Outstanding quality and service. Will buy again!",
        "Very poor experience. Many issues encountered.",
        "Adequate for basic needs, but lacks advanced features.",
        "Couldn't be happier with this purchase.",
        "Avoid at all costs. Terrible quality.",
        "Average performance. Neither good nor bad.",
        "Top-notch quality and design. Very impressed!",
    ];

    let test_labels = vec![
        "positive", "negative", "neutral", "positive", "negative", "neutral", "positive",
        "negative", "neutral", "positive",
    ];

    // Convert to strings
    let train_texts = train_texts.iter().map(|t| t.to_string()).collect();
    let train_labels = train_labels.iter().map(|l| l.to_string()).collect();
    let test_texts = test_texts.iter().map(|t| t.to_string()).collect();
    let test_labels = test_labels.iter().map(|l| l.to_string()).collect();

    // Create datasets
    let train_dataset = TextDataset::new(train_texts, train_labels)?;
    let test_dataset = TextDataset::new(test_texts, test_labels)?;

    Ok((train_dataset, test_dataset))
}
