//! Example of automated model selection based on multiple metrics
//!
//! This example demonstrates how to use the model selection framework
//! to automatically choose the best model from a set of candidates.

use scirs2_metrics::selection::{AggregationStrategy, ModelSelectionBuilder, ModelSelector};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Automated Model Selection Example");
    println!("================================");

    // Create model evaluation results
    let model_scores = create_model_scores();

    // Display all model scores
    println!("\nModel Evaluation Results:");
    println!("------------------------");
    for (model_name, scores) in &model_scores {
        println!("{}:", model_name);
        for (metric, value) in scores {
            println!("  {}: {:.4}", metric, value);
        }
        println!();
    }

    // Example 1: Basic weighted selection
    println!("1. Basic Weighted Selection");
    println!("---------------------------");

    let mut selector = ModelSelector::new();
    selector
        .add_metric("accuracy", 0.4, true)      // 40% weight, higher is better
        .add_metric("precision", 0.3, true)     // 30% weight, higher is better
        .add_metric("inference_time", 0.3, false); // 30% weight, lower is better (faster)

    let best_model = selector.select_best(&model_scores)?;
    let rankings = selector.rank_models(&model_scores)?;

    println!("Best model: {}", best_model);
    println!("\nComplete rankings:");
    for (i, (model, score)) in rankings.iter().enumerate() {
        println!("{}: {} (score: {:.4})", i + 1, model, score);
    }

    // Example 2: Selection with thresholds
    println!("\n2. Selection with Minimum Thresholds");
    println!("------------------------------------");

    let mut threshold_selector = ModelSelector::new();
    threshold_selector
        .add_metric("accuracy", 0.6, true)
        .add_metric("f1_score", 0.4, true)
        .add_threshold("accuracy", 0.85)        // Minimum 85% accuracy required
        .add_threshold("f1_score", 0.80); // Minimum 80% F1 score required

    match threshold_selector.select_best(&model_scores) {
        Ok(best) => {
            let rankings = threshold_selector.rank_models(&model_scores)?;
            println!("Best model meeting thresholds: {}", best);
            println!("Models meeting thresholds: {}", rankings.len());
            for (model, score) in &rankings {
                println!("  {}: {:.4}", model, score);
            }
        }
        Err(e) => {
            println!("No models meet the threshold requirements: {}", e);
            let all_rankings = threshold_selector.rank_models(&model_scores)?;
            println!(
                "Models that would qualify without thresholds: {}",
                all_rankings.len()
            );
        }
    }

    // Example 3: Pareto optimal selection
    println!("\n3. Pareto Optimal Models");
    println!("------------------------");

    let mut pareto_selector = ModelSelector::new();
    pareto_selector
        .add_metric("accuracy", 1.0, true)
        .add_metric("inference_time", 1.0, false)  // Consider accuracy vs speed trade-off
        .add_metric("memory_usage", 1.0, false); // And memory efficiency

    let pareto_optimal = pareto_selector.find_pareto_optimal(&model_scores);
    println!("Pareto optimal models (not dominated by others):");
    for model in &pareto_optimal {
        println!("  {}", model);
        if let Some(scores) = model_scores.get(model) {
            for (metric, value) in scores {
                println!("    {}: {:.4}", metric, value);
            }
        }
    }

    // Example 4: Different aggregation strategies
    println!("\n4. Comparison of Aggregation Strategies");
    println!("---------------------------------------");

    let strategies = [
        ("Weighted Sum", AggregationStrategy::WeightedSum),
        ("Geometric Mean", AggregationStrategy::WeightedGeometricMean),
        ("Harmonic Mean", AggregationStrategy::WeightedHarmonicMean),
        ("Conservative (Min)", AggregationStrategy::MinScore),
        ("Optimistic (Max)", AggregationStrategy::MaxScore),
    ];

    for (name, strategy) in &strategies {
        let mut strategy_selector = ModelSelector::new();
        strategy_selector
            .add_metric("accuracy", 0.5, true)
            .add_metric("f1_score", 0.5, true)
            .with_aggregation(*strategy);

        if let Ok(best) = strategy_selector.select_best(&model_scores) {
            println!("{}: {}", name, best);
        }
    }

    // Example 5: Builder pattern for complex selection
    println!("\n5. Complex Selection using Builder Pattern");
    println!("------------------------------------------");

    let result = ModelSelectionBuilder::new()
        .metric("accuracy", 0.3, true)
        .metric("precision", 0.2, true)
        .metric("recall", 0.2, true)
        .metric("inference_time", 0.2, false)
        .metric("memory_usage", 0.1, false)
        .threshold("accuracy", 0.80)
        .threshold("precision", 0.75)
        .aggregation(AggregationStrategy::WeightedSum)
        .select(&model_scores)?;

    println!("{}", result);

    // Example 6: Domain-specific selection scenarios
    println!("\n6. Domain-Specific Selection Scenarios");
    println!("--------------------------------------");

    // Production deployment: prioritize speed and memory
    println!("\nProduction Deployment (Speed + Memory Priority):");
    let production_best = ModelSelectionBuilder::new()
        .metric("accuracy", 0.3, true)
        .metric("inference_time", 0.4, false)
        .metric("memory_usage", 0.3, false)
        .threshold("accuracy", 0.75)  // Minimum acceptable accuracy
        .select(&model_scores)?;
    println!("Best for production: {}", production_best.selected_model);

    // Research setting: prioritize accuracy
    println!("\nResearch Setting (Accuracy Priority):");
    let research_best = ModelSelectionBuilder::new()
        .metric("accuracy", 0.5, true)
        .metric("f1_score", 0.3, true)
        .metric("precision", 0.2, true)
        .select(&model_scores)?;
    println!("Best for research: {}", research_best.selected_model);

    // Balanced approach
    println!("\nBalanced Approach (All metrics equally weighted):");
    let balanced_best = ModelSelectionBuilder::new()
        .metric("accuracy", 0.25, true)
        .metric("precision", 0.25, true)
        .metric("inference_time", 0.25, false)
        .metric("memory_usage", 0.25, false)
        .aggregation(AggregationStrategy::WeightedGeometricMean)
        .select(&model_scores)?;
    println!("Best balanced model: {}", balanced_best.selected_model);

    // Example 7: Handling edge cases
    println!("\n7. Edge Cases and Error Handling");
    println!("--------------------------------");

    // Empty model set
    let empty_scores = HashMap::new();
    let empty_selector = ModelSelector::new();
    match empty_selector.select_best(&empty_scores) {
        Ok(_) => println!("Unexpected success with empty model set"),
        Err(e) => println!("Expected error with empty models: {}", e),
    }

    // Missing metrics
    let mut incomplete_scores = HashMap::new();
    incomplete_scores.insert("incomplete_model".to_string(), vec![("accuracy", 0.9)]); // Missing other metrics

    let mut strict_selector = ModelSelector::new();
    strict_selector
        .add_metric("accuracy", 0.5, true)
        .add_metric("missing_metric", 0.5, true);

    match strict_selector.select_best(&incomplete_scores) {
        Ok(best) => println!("Selected model despite missing metrics: {}", best),
        Err(e) => println!("Error with missing metrics: {}", e),
    }

    println!("\nModel selection example completed successfully!");
    Ok(())
}

/// Creates a comprehensive set of model evaluation results
fn create_model_scores() -> HashMap<String, Vec<(&'static str, f64)>> {
    let mut scores = HashMap::new();

    // High-accuracy but slow model (e.g., large ensemble)
    scores.insert(
        "ensemble_model".to_string(),
        vec![
            ("accuracy", 0.92),
            ("precision", 0.90),
            ("recall", 0.89),
            ("f1_score", 0.895),
            ("inference_time", 150.0), // milliseconds (slower)
            ("memory_usage", 512.0),   // MB (high memory)
            ("training_time", 3600.0), // seconds
        ],
    );

    // Balanced model (good trade-off)
    scores.insert(
        "random_forest".to_string(),
        vec![
            ("accuracy", 0.88),
            ("precision", 0.86),
            ("recall", 0.84),
            ("f1_score", 0.85),
            ("inference_time", 45.0), // milliseconds (moderate)
            ("memory_usage", 128.0),  // MB (moderate memory)
            ("training_time", 600.0), // seconds
        ],
    );

    // Fast but less accurate model (e.g., simple model)
    scores.insert(
        "logistic_regression".to_string(),
        vec![
            ("accuracy", 0.82),
            ("precision", 0.81),
            ("recall", 0.80),
            ("f1_score", 0.805),
            ("inference_time", 5.0), // milliseconds (very fast)
            ("memory_usage", 16.0),  // MB (low memory)
            ("training_time", 30.0), // seconds
        ],
    );

    // High precision but lower recall model
    scores.insert(
        "svm_model".to_string(),
        vec![
            ("accuracy", 0.85),
            ("precision", 0.92), // Very high precision
            ("recall", 0.78),    // Lower recall
            ("f1_score", 0.845),
            ("inference_time", 80.0),  // milliseconds (moderate-slow)
            ("memory_usage", 256.0),   // MB (moderate-high memory)
            ("training_time", 1200.0), // seconds
        ],
    );

    // High recall but lower precision model
    scores.insert(
        "naive_bayes".to_string(),
        vec![
            ("accuracy", 0.83),
            ("precision", 0.79), // Lower precision
            ("recall", 0.91),    // Very high recall
            ("f1_score", 0.848),
            ("inference_time", 12.0), // milliseconds (fast)
            ("memory_usage", 32.0),   // MB (low memory)
            ("training_time", 60.0),  // seconds
        ],
    );

    // Neural network model (good accuracy, variable performance)
    scores.insert(
        "neural_network".to_string(),
        vec![
            ("accuracy", 0.90),
            ("precision", 0.88),
            ("recall", 0.87),
            ("f1_score", 0.875),
            ("inference_time", 25.0),  // milliseconds (fast with GPU)
            ("memory_usage", 384.0),   // MB (high memory)
            ("training_time", 7200.0), // seconds (long training)
        ],
    );

    // Gradient boosting model
    scores.insert(
        "gradient_boosting".to_string(),
        vec![
            ("accuracy", 0.91),
            ("precision", 0.89),
            ("recall", 0.88),
            ("f1_score", 0.885),
            ("inference_time", 35.0),  // milliseconds
            ("memory_usage", 192.0),   // MB
            ("training_time", 1800.0), // seconds
        ],
    );

    scores
}
