//! Example of domain-specific metric collections
//!
//! This example demonstrates how to use the specialized metric suites for
//! different machine learning domains.

use scirs2_core::ndarray::Array1;
use scirs2_metrics::domains::*;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Domain-Specific Metrics Example");
    println!("==============================");

    // Create a domain suite with all available domain metrics
    let domain_suite = create_domain_suite();

    println!("\nAvailable domains:");
    for domain in domain_suite.available_domains() {
        println!("  - {domain}");
    }

    // Example 1: Computer Vision - Object Detection
    println!("\n1. Computer Vision - Object Detection");
    println!("------------------------------------");

    computer_vision_example()?;

    // Example 2: Natural Language Processing - Text Classification
    println!("\n2. Natural Language Processing - Text Classification");
    println!("--------------------------------------------------");

    nlp_example()?;

    // Example 3: Time Series - Forecasting
    println!("\n3. Time Series - Forecasting");
    println!("---------------------------");

    time_series_example()?;

    // Example 4: Recommender Systems - Ranking
    println!("\n4. Recommender Systems - Ranking");
    println!("-------------------------------");

    recommender_example()?;

    // Example 5: Anomaly Detection
    println!("\n5. Anomaly Detection");
    println!("------------------");

    anomaly_detection_example()?;

    // Example 6: Cross-Domain Evaluation
    println!("\n6. Cross-Domain Evaluation Suite");
    println!("-------------------------------");

    cross_domain_example();

    println!("\nDomain-specific metrics example completed successfully!");
    Ok(())
}

/// Computer Vision - Object Detection Example
#[allow(dead_code)]
fn computer_vision_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use scirs2_metrics::domains::computer_vision::*;

    let cv_metrics = ObjectDetectionMetrics::new().with_confidence_threshold(0.5);

    // Example predictions: (x1, y1, x2, y2, confidence, class_id)
    let predictions = vec![
        (10.0, 10.0, 50.0, 50.0, 0.9, 1),   // High confidence detection
        (60.0, 60.0, 100.0, 100.0, 0.7, 2), // Medium confidence detection
        (20.0, 20.0, 40.0, 40.0, 0.3, 1),   // Low confidence (filtered out)
    ];

    // Ground truth: (x1, y1, x2, y2, class_id)
    let ground_truth = vec![
        (12.0, 12.0, 48.0, 48.0, 1),     // Close to first prediction
        (70.0, 70.0, 110.0, 110.0, 2),   // Close to second prediction
        (200.0, 200.0, 220.0, 220.0, 3), // Missed detection
    ];

    let results = cv_metrics.evaluate_object_detection(&predictions, &ground_truth, 0.5)?;

    println!("Object Detection Results:");
    println!("  mAP@0.5: {:.4}", results.map);
    println!("  Precision: {:.4}", results.precision);
    println!("  Recall: {:.4}", results.recall);
    println!("  F1 Score: {:.4}", results.f1_score);
    println!("  Per-class AP: {:?}", results.per_class_ap);

    Ok(())
}

/// Natural Language Processing - Text Classification Example
#[allow(dead_code)]
fn nlp_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use scirs2_metrics::domains::nlp::*;

    // Text Generation Example
    let text_gen_metrics = TextGenerationMetrics::new();

    let references = vec![
        "The cat sat on the mat".to_string(),
        "A quick brown fox jumps over the lazy dog".to_string(),
    ];

    let candidates = vec![
        "The cat sits on the mat".to_string(),
        "A quick brown fox jumped over the lazy dog".to_string(),
    ];

    let gen_results = text_gen_metrics.evaluate_generation(&references, &candidates)?;

    println!("Text Generation Results:");
    println!("  BLEU-1: {:.4}", gen_results.bleu_1);
    println!("  BLEU-4: {:.4}", gen_results.bleu_4);
    println!("  ROUGE-L: {:.4}", gen_results.rouge_l);
    println!("  METEOR: {:.4}", gen_results.meteor);

    // Text Classification Example
    let classification_metrics = TextClassificationMetrics::new();

    let y_true = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 1, 0, 2]);
    let y_pred = Array1::from_vec(vec![0, 2, 1, 0, 1, 2, 1, 1, 2]);

    let class_results = classification_metrics.evaluate_classification(&y_true, &y_pred)?;

    println!("Text Classification Results:");
    println!("  Accuracy: {:.4}", class_results.accuracy);
    println!("  Macro F1: {:.4}", class_results.macro_f1);
    println!("  Weighted F1: {:.4}", class_results.weighted_f1);

    Ok(())
}

/// Time Series - Forecasting Example
#[allow(dead_code)]
fn time_series_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use scirs2_metrics::domains::time_series::*;

    let forecasting_metrics = ForecastingMetrics::new();

    // Simulate time series forecast
    let y_true = Array1::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    let y_pred = Array1::from_vec(vec![10.2, 10.8, 12.1, 12.9, 14.2, 14.8]);
    let y_train = Array1::from_vec(vec![8.0, 9.0, 10.0]); // Historical data

    let results = forecasting_metrics.evaluate_forecast(&y_true, &y_pred, Some(&y_train))?;

    println!("Forecasting Results:");
    println!("  MAE: {:.4}", results.mae);
    println!("  RMSE: {:.4}", results.rmse);
    println!("  MAPE: {:.4}%", results.mape);
    println!("  SMAPE: {:.4}%", results.smape);
    println!("  MASE: {:.4}", results.mase);
    println!(
        "  Directional Accuracy: {:.4}",
        results.directional_accuracy
    );
    println!("  Forecast Bias: {:.4}", results.forecast_bias);

    // Time Series Anomaly Detection
    let ts_anomaly_metrics =
        anomaly_detection::TimeSeriesAnomalyMetrics::new().with_tolerance_window(2);

    let y_true_anomaly = Array1::from_vec(vec![0, 0, 1, 1, 0, 0, 1, 0, 0]);
    let y_pred_anomaly = Array1::from_vec(vec![0, 0, 0, 1, 0, 1, 1, 0, 0]);
    let timestamps = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let anomaly_results = ts_anomaly_metrics.evaluate_time_series_anomalies(
        &y_true_anomaly,
        &y_pred_anomaly,
        Some(&timestamps),
    )?;

    println!("Time Series Anomaly Detection:");
    println!("  Precision: {:.4}", anomaly_results.precision);
    println!("  Recall: {:.4}", anomaly_results.recall);
    println!(
        "  Point-Adjust Precision: {:.4}",
        anomaly_results.point_adjust_precision
    );
    println!(
        "  Point-Adjust Recall: {:.4}",
        anomaly_results.point_adjust_recall
    );
    println!("  NAB Score: {:.4}", anomaly_results.nab_score);
    println!("  Average Delay: {:.4}", anomaly_results.average_delay);

    Ok(())
}

/// Recommender Systems - Ranking Example
#[allow(dead_code)]
fn recommender_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use scirs2_metrics::domains::recommender::*;

    let ranking_metrics = RecommenderRankingMetrics::new()
        .with_k_values(vec![5, 10])
        .with_total_items(100);

    // Example: 2 users, 5 items each
    let y_true = vec![
        Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]), // User 1 relevance
        Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]), // User 2 relevance
    ];

    let y_score = vec![
        Array1::from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7]), // User 1 scores
        Array1::from_vec(vec![0.3, 0.9, 0.4, 0.8, 0.1]), // User 2 scores
    ];

    let recommended_items = vec![
        vec![0, 2, 4, 1, 3], // User 1 recommendations (sorted by score)
        vec![1, 3, 2, 0, 4], // User 2 recommendations
    ];

    let results = ranking_metrics.evaluate_ranking(&y_true, &y_score, &recommended_items)?;

    println!("Recommender Ranking Results:");
    for (&k, &ndcg) in &results.ndcg_at_k {
        println!("  NDCG@{k}: {ndcg:.4}");
    }
    for (&k, &precision) in &results.precision_at_k {
        println!("  Precision@{k}: {precision:.4}");
    }
    for (&k, &recall) in &results.recall_at_k {
        println!("  Recall@{k}: {recall:.4}");
    }
    println!("  MAP: {:.4}", results.map);
    println!("  MRR: {:.4}", results.mrr);

    // Rating Prediction Example
    let rating_metrics = RatingPredictionMetrics::new().with_thresholds(vec![3.0, 4.0]);

    let y_true_ratings = Array1::from_vec(vec![4.0, 3.5, 2.0, 4.5, 1.0, 5.0]);
    let y_pred_ratings = Array1::from_vec(vec![3.8, 3.2, 2.3, 4.2, 1.5, 4.8]);

    let rating_results =
        rating_metrics.evaluate_rating_prediction(&y_true_ratings, &y_pred_ratings)?;

    println!("Rating Prediction Results:");
    println!("  RMSE: {:.4}", rating_results.rmse);
    println!("  MAE: {:.4}", rating_results.mae);
    for (threshold, precision) in &rating_results.precision_at_threshold {
        println!("  Precision@{threshold}: {precision:.4}");
    }

    Ok(())
}

/// Anomaly Detection Example
#[allow(dead_code)]
fn anomaly_detection_example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use scirs2_metrics::domains::anomaly_detection::*;

    // Basic Detection Metrics
    let detection_metrics = DetectionMetrics::new().with_threshold(0.5);

    let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let y_score = Array1::from_vec(vec![0.1, 0.2, 0.9, 0.3, 0.8, 0.1, 0.7, 0.2]);

    let detection_results = detection_metrics.evaluate_detection(&y_true, &y_score)?;

    println!("Anomaly Detection Results:");
    println!("  Accuracy: {:.4}", detection_results.accuracy);
    println!("  Precision: {:.4}", detection_results.precision);
    println!("  Recall: {:.4}", detection_results.recall);
    println!("  F1 Score: {:.4}", detection_results.f1_score);
    println!("  AUC-ROC: {:.4}", detection_results.auc_roc);
    println!("  AUC-PR: {:.4}", detection_results.auc_pr);
    println!(
        "  False Alarm Rate: {:.4}",
        detection_results.false_alarm_rate
    );
    println!(
        "  Miss Detection Rate: {:.4}",
        detection_results.miss_detection_rate
    );
    println!("  MCC: {:.4}", detection_results.mcc);

    // Distribution-based Metrics
    let distribution_metrics = DistributionMetrics::new();

    let normal_samples = Array1::from_vec(vec![0.0, 0.1, 0.2, 0.1, 0.0, -0.1, 0.05]);
    let anomaly_samples = Array1::from_vec(vec![2.0, 2.5, 3.0, 2.8, 2.2, 2.7]);

    let dist_results =
        distribution_metrics.evaluate_distribution_anomalies(&normal_samples, &anomaly_samples)?;

    println!("Distribution Analysis:");
    println!("  KL Divergence: {:.4}", dist_results.kl_divergence);
    println!("  JS Divergence: {:.4}", dist_results.js_divergence);
    println!(
        "  Wasserstein Distance: {:.4}",
        dist_results.wasserstein_distance
    );
    println!("  MMD: {:.4}", dist_results.mmd);
    println!("  Energy Distance: {:.4}", dist_results.energy_distance);
    println!("  KS Statistic: {:.4}", dist_results.ks_statistic);

    Ok(())
}

/// Cross-Domain Evaluation Example
#[allow(dead_code)]
fn cross_domain_example() {
    let suite = create_domain_suite();

    println!("Cross-Domain Evaluation Summary:");
    println!("Available domains: {}", suite.available_domains().len());

    // Show available metrics for each domain
    println!("\nDomain capabilities:");

    // Computer Vision
    let cv_suite = suite.computer_vision();
    println!(
        "  {}: {} metrics",
        cv_suite.domain_name(),
        cv_suite.available_metrics().len()
    );

    // NLP
    let nlp_suite = suite.nlp();
    println!(
        "  {}: {} metrics",
        nlp_suite.domain_name(),
        nlp_suite.available_metrics().len()
    );

    // Time Series
    let ts_suite = suite.time_series();
    println!(
        "  {}: {} metrics",
        ts_suite.domain_name(),
        ts_suite.available_metrics().len()
    );

    // Recommender Systems
    let rec_suite = suite.recommender();
    println!(
        "  {}: {} metrics",
        rec_suite.domain_name(),
        rec_suite.available_metrics().len()
    );

    // Anomaly Detection
    let ad_suite = suite.anomaly_detection();
    println!(
        "  {}: {} metrics",
        ad_suite.domain_name(),
        ad_suite.available_metrics().len()
    );

    println!("\nExample usage patterns:");
    println!("  - Use cv_suite.object_detection() for object detection tasks");
    println!("  - Use nlp_suite.text_generation() for text generation evaluation");
    println!("  - Use ts_suite.forecasting() for time series prediction");
    println!("  - Use rec_suite.ranking() for recommendation ranking");
    println!("  - Use ad_suite.detection() for general anomaly detection");

    println!("\nEach domain provides:");
    println!("  - Pre-configured metric suites for common tasks");
    println!("  - Domain-specific evaluation workflows");
    println!("  - Comprehensive result structures");
    println!("  - Best practices and metric selection guidance");
}
