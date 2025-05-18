//! Machine learning integration example

use scirs2_text::{
    BatchTextProcessor, FeatureExtractionMode, MLTextPreprocessor, TextDataset, TextMLPipeline,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Machine Learning Integration Demo");
    println!("================================\n");

    // Sample dataset for demonstration
    let texts = vec![
        "This product is absolutely amazing! I love it.",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special but works fine.",
        "Excellent quality and fast shipping.",
        "Complete waste of money, very disappointed.",
        "Good value for the price, satisfied with purchase.",
        "Outstanding service and great product!",
        "Not worth it, many issues with this item.",
    ];

    let labels = vec![
        "positive", "negative", "neutral", "positive", "negative", "positive", "positive",
        "negative",
    ];

    // Create dataset
    let dataset = TextDataset::new(
        texts.iter().map(|s| s.to_string()).collect(),
        labels.iter().map(|s| s.to_string()).collect(),
    )?;

    // Demonstrate different feature extraction modes
    println!("1. TF-IDF Feature Extraction");
    println!("---------------------------");

    let mut tfidf_processor = MLTextPreprocessor::new(FeatureExtractionMode::TfIdf)
        .with_tfidf_params(0.1, 0.9, Some(100));

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
    let tfidf_features = tfidf_processor.fit_transform(&text_refs)?;

    println!(
        "TF-IDF Features shape: {:?}",
        tfidf_features.features.shape()
    );
    println!(
        "First document features (first 5 values): {:?}\n",
        &tfidf_features
            .features
            .row(0)
            .iter()
            .take(5)
            .collect::<Vec<_>>()
    );

    // Topic modeling features
    println!("2. Topic Modeling Features");
    println!("-------------------------");

    let mut topic_processor =
        MLTextPreprocessor::new(FeatureExtractionMode::TopicModeling).with_topic_modeling(3);

    let topic_features = topic_processor.fit_transform(&text_refs)?;

    println!(
        "Topic Features shape: {:?}",
        topic_features.features.shape()
    );
    println!(
        "Topic distribution for first document: {:?}\n",
        topic_features.features.row(0)
    );

    // Combined features
    println!("3. Combined Features");
    println!("-------------------");

    let mut combined_processor = MLTextPreprocessor::new(FeatureExtractionMode::Combined);
    let combined_features = combined_processor.fit_transform(&text_refs)?;

    println!(
        "Combined Features shape: {:?}",
        combined_features.features.shape()
    );
    println!("Metadata: {:?}\n", combined_features.metadata);

    // ML Pipeline
    println!("4. ML Pipeline with Classification");
    println!("---------------------------------");

    let mut pipeline = TextMLPipeline::with_mode(FeatureExtractionMode::TfIdf)
        .configure_preprocessor(|p| {
            p.with_tfidf_params(0.0, 1.0, Some(50))
                .with_feature_selection(20)
        });

    let features = pipeline.process(&text_refs)?;
    println!("Pipeline features shape: {:?}", features.features.shape());

    // Batch processing for large datasets
    println!("\n5. Batch Processing");
    println!("-------------------");

    let mut batch_processor = BatchTextProcessor::new(3);
    let batches = batch_processor.process_batches(&text_refs)?;

    println!("Number of batches: {}", batches.len());
    for (i, batch) in batches.iter().enumerate() {
        println!("Batch {} shape: {:?}", i + 1, batch.features.shape());
    }

    // Feature extraction for classification
    println!("\n6. Classification with ML Features");
    println!("----------------------------------");

    // Split data
    let (train_dataset, test_dataset) = dataset.train_test_split(0.25, Some(42))?;

    // Extract features
    let train_texts: Vec<&str> = train_dataset.texts.iter().map(|s| s.as_ref()).collect();
    let test_texts: Vec<&str> = test_dataset.texts.iter().map(|s| s.as_ref()).collect();

    let mut feature_extractor = MLTextPreprocessor::new(FeatureExtractionMode::TfIdf);
    feature_extractor.fit(&train_texts)?;

    let train_features = feature_extractor.transform(&train_texts)?;
    let test_features = feature_extractor.transform(&test_texts)?;

    println!("Training features: {:?}", train_features.features.shape());
    println!("Test features: {:?}", test_features.features.shape());

    // In a real scenario, you would now use these features with a classifier
    println!("\nFeatures are ready for machine learning models!");

    // Demonstrate feature statistics
    println!("\n7. Feature Statistics");
    println!("--------------------");

    let feature_means = train_features.features.mean_axis(ndarray::Axis(0)).unwrap();
    let feature_stds = train_features.features.std_axis(ndarray::Axis(0), 0.0);

    println!(
        "Mean of first 5 features: {:?}",
        &feature_means.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "Std of first 5 features: {:?}",
        &feature_stds.iter().take(5).collect::<Vec<_>>()
    );

    Ok(())
}
