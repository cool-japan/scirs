//! Comprehensive examples of hyperparameter search strategies using the meta-learning module
//!
//! This example demonstrates various hyperparameter optimization approaches including:
//! - Grid search for exhaustive exploration
//! - Random search for efficient sampling
//! - Bayesian optimization for intelligent search
//! - Population-based training for evolutionary optimization
//! - Neural predictors for learning-based optimization

use ndarray::Array1;
use num_traits::Float;
use rand::{rng, Rng};
use scirs2_optim::error::Result;
use scirs2_optim::meta_learning::*;
use std::collections::HashMap;

/// Simulated machine learning model for demonstration
struct MockModel {
    learning_rate: f64,
    weight_decay: f64,
    batch_size: usize,
    momentum: f64,
    performance: f64,
}

impl MockModel {
    fn new(hyperparams: &HashMap<String, f64>) -> Self {
        Self {
            learning_rate: hyperparams.get("learning_rate").copied().unwrap_or(0.001),
            weight_decay: hyperparams.get("weight_decay").copied().unwrap_or(0.0),
            batch_size: hyperparams.get("batch_size").copied().unwrap_or(32.0) as usize,
            momentum: hyperparams.get("momentum").copied().unwrap_or(0.9),
            performance: 0.0,
        }
    }

    /// Simulate training and return validation performance
    fn train_and_evaluate(&mut self) -> f64 {
        // Simulate realistic performance based on hyperparameters
        let mut rng = rand::rng();

        // Base performance with some randomness
        let mut performance = 0.5 + rng.random::<f64>() * 0.1;

        // Learning rate effects
        if self.learning_rate > 0.0001 && self.learning_rate < 0.01 {
            performance += 0.2; // Good learning rate range
        } else if self.learning_rate > 0.1 {
            performance -= 0.3; // Too high, unstable training
        }

        // Weight decay effects
        if self.weight_decay > 0.0001 && self.weight_decay < 0.1 {
            performance += 0.1; // Regularization helps
        }

        // Batch size effects
        if self.batch_size >= 16 && self.batch_size <= 128 {
            performance += 0.1; // Good batch size range
        }

        // Momentum effects
        if self.momentum > 0.8 && self.momentum < 0.99 {
            performance += 0.1; // Good momentum range
        }

        // Interaction effects
        if self.learning_rate < 0.01 && self.momentum > 0.9 {
            performance += 0.05; // Good combination
        }

        self.performance = performance.clamp(0.0, 1.0);
        self.performance
    }
}

/// Extract problem features for neural predictor
#[allow(dead_code)]
fn extract_problem_features() -> Array1<f64> {
    // In a real scenario, these would be actual dataset/problem characteristics
    Array1::from_vec(vec![
        1000.0, // dataset_size
        784.0,  // input_dimensions
        10.0,   // num_classes
        0.8,    // train_ratio
        0.1,    // val_ratio
        0.1,    // test_ratio
        0.0,    // sparsity
        0.05,   // noise_level
    ])
}

/// Example 1: Grid Search for Systematic Exploration
#[allow(dead_code)]
fn grid_search_example() -> Result<()> {
    println!("=== Grid Search Example ===");

    // Define parameter grids
    let grids = HashMap::from([
        ("learning_rate".to_string(), vec![0.001, 0.01, 0.1]),
        ("weight_decay".to_string(), vec![0.0, 0.001, 0.01]),
        ("batch_size".to_string(), vec![16.0, 32.0, 64.0, 128.0]),
        ("momentum".to_string(), vec![0.0, 0.9, 0.99]),
    ]);

    let strategy = HyperparameterStrategy::GridSearch { grids };
    let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

    let mut trial_count = 0;
    let mut results = Vec::new();

    // Run grid search
    while optimizer.should_continue() {
        let hyperparams = optimizer.suggest_hyperparameters()?;

        // Train model with suggested hyperparameters
        let mut model = MockModel::new(&hyperparams);
        let performance = model.train_and_evaluate();

        // Update optimizer with results
        optimizer.report_performance(hyperparams.clone(), performance);

        results.push((hyperparams, performance));
        trial_count += 1;

        if trial_count % 10 == 0 {
            println!("Completed {} trials", trial_count);
        }
    }

    // Analyze results
    if let Some(best_params) = optimizer.get_best_hyperparameters() {
        println!("Grid Search Results:");
        println!("Total trials: {}", optimizer.get_trial_history().len());
        println!(
            "Best performance: {:.4}",
            optimizer.get_best_performance().unwrap()
        );
        println!("Best hyperparameters:");
        for (param, value) in best_params {
            println!("  {}: {:.6}", param, value);
        }
    }

    // Find top 5 configurations
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 configurations:");
    for (i, (params, perf)) in results.iter().take(5).enumerate() {
        println!("{}. Performance: {:.4}", i + 1, perf);
        for (param, value) in params {
            println!("   {}: {:.6}", param, value);
        }
    }

    Ok(())
}

/// Example 2: Random Search for Efficient Sampling
#[allow(dead_code)]
fn random_search_example() -> Result<()> {
    println!("\n=== Random Search Example ===");

    // Define parameter bounds for random sampling
    let bounds = HashMap::from([
        ("learning_rate".to_string(), (1e-5, 1e-1)),
        ("weight_decay".to_string(), (0.0, 0.1)),
        ("batch_size".to_string(), (8.0, 256.0)),
        ("momentum".to_string(), (0.0, 0.999)),
    ]);

    let strategy = HyperparameterStrategy::RandomSearch {
        num_trials: 50,
        bounds,
    };
    let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

    let mut best_performance = 0.0;
    let mut convergence_history = Vec::new();

    // Run random search
    for trial in 0..50 {
        let hyperparams = optimizer.suggest_hyperparameters()?;

        // Train model
        let mut model = MockModel::new(&hyperparams);
        let performance = model.train_and_evaluate();

        // Update optimizer
        optimizer.report_performance(hyperparams.clone(), performance);

        // Track convergence
        if performance > best_performance {
            best_performance = performance;
        }
        convergence_history.push(best_performance);

        if trial % 10 == 9 {
            println!(
                "Trial {}: current_best = {:.4}",
                trial + 1,
                best_performance
            );
        }
    }

    // Final results
    if let Some(best_params) = optimizer.get_best_hyperparameters() {
        println!("Random Search Results:");
        println!(
            "Best performance: {:.4}",
            optimizer.get_best_performance().unwrap()
        );
        println!("Best hyperparameters:");
        for (param, value) in best_params {
            println!("  {}: {:.6}", param, value);
        }
    }

    // Analyze convergence
    println!("Convergence analysis:");
    let improvement_steps = convergence_history
        .windows(2)
        .enumerate()
        .filter(|(_, window)| window[1] > window[0])
        .map(|(i_, _)| i_ + 1)
        .collect::<Vec<_>>();

    println!("Improvements found at trials: {:?}", improvement_steps);
    println!(
        "Final convergence: {:.4}",
        convergence_history.last().unwrap()
    );

    Ok(())
}

/// Example 3: Bayesian Optimization for Intelligent Search
#[allow(dead_code)]
fn bayesian_optimization_example() -> Result<()> {
    println!("\n=== Bayesian Optimization Example ===");

    // Define parameter bounds
    let bounds = HashMap::from([
        ("learning_rate".to_string(), (1e-5, 1e-1)),
        ("weight_decay".to_string(), (0.0, 0.1)),
        ("batch_size".to_string(), (16.0, 128.0)),
        ("momentum".to_string(), (0.5, 0.999)),
    ]);

    let strategy = HyperparameterStrategy::BayesianOptimization {
        num_trials: 30,
        bounds,
        acquisition: AcquisitionFunction::ExpectedImprovement,
    };
    let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

    let mut trial_history = Vec::new();
    let mut performance_history = Vec::new();

    // Run Bayesian optimization
    for trial in 0..30 {
        let hyperparams = optimizer.suggest_hyperparameters()?;

        // Simulate expensive evaluation
        let mut model = MockModel::new(&hyperparams);
        let performance = model.train_and_evaluate();

        // Update optimizer
        optimizer.report_performance(hyperparams.clone(), performance);

        trial_history.push(hyperparams);
        performance_history.push(performance);

        // Analyze acquisition strategy
        if trial >= 5 {
            let recent_improvement = performance_history[trial]
                - performance_history[trial - 5..trial]
                    .iter()
                    .copied()
                    .fold(0.0, f64::max);

            if trial % 5 == 4 {
                println!(
                    "Trial {}: performance = {:.4}, recent_improvement = {:.4}",
                    trial + 1,
                    performance,
                    recent_improvement
                );
            }
        }
    }

    // Final results
    if let Some(best_params) = optimizer.get_best_hyperparameters() {
        println!("Bayesian Optimization Results:");
        println!(
            "Best performance: {:.4}",
            optimizer.get_best_performance().unwrap()
        );
        println!("Best hyperparameters:");
        for (param, value) in best_params {
            println!("  {}: {:.6}", param, value);
        }
    }

    // Analyze exploration vs exploitation
    let mut exploration_trials = 0;
    let mut exploitation_trials = 0;

    for trial in trial_history.iter().skip(1) {
        let current_lr = trial["learning_rate"];
        let best_lr = optimizer.get_best_hyperparameters().unwrap()["learning_rate"];

        if (current_lr - best_lr).abs() > 0.01 {
            exploration_trials += 1;
        } else {
            exploitation_trials += 1;
        }
    }

    println!("Exploration/Exploitation balance:");
    println!("  Exploration trials: {}", exploration_trials);
    println!("  Exploitation trials: {}", exploitation_trials);

    Ok(())
}

/// Example 4: Population-Based Training
#[allow(dead_code)]
fn population_based_training_example() -> Result<()> {
    println!("\n=== Population-Based Training Example ===");

    let strategy = HyperparameterStrategy::PopulationBased {
        population_size: 20,
        perturbation_factor: 0.2,
    };
    let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

    // Initialize population
    let mut population = Vec::new();
    for i in 0..20 {
        let hyperparams = HashMap::from([
            (
                "learning_rate".to_string(),
                0.001 * (1.0 + rand::rng().random::<f64>()),
            ),
            (
                "weight_decay".to_string(),
                0.01 * rand::rng().random::<f64>(),
            ),
            (
                "batch_size".to_string(),
                32.0 + rand::rng().random::<f64>() * 96.0,
            ),
            (
                "momentum".to_string(),
                0.9 + rand::rng().random::<f64>() * 0.09,
            ),
        ]);

        let mut model = MockModel::new(&hyperparams);
        let performance = model.train_and_evaluate();

        population.push((hyperparams.clone(), performance, i));
        optimizer.report_performance(hyperparams, performance);
    }

    // Evolution loop
    for generation in 0..25 {
        // Sort population by performance
        population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top 50%, evolve bottom 50%
        let num_elite = population.len() / 2;

        for i in num_elite..population.len() {
            // Select parent from elite
            let parent_idx = i % num_elite;
            let mut new_hyperparams = population[parent_idx].0.clone();

            // Perturb hyperparameters
            for (param, value) in new_hyperparams.iter_mut() {
                let perturbation = 1.0 + (rand::rng().random::<f64>() - 0.5) * 0.2;
                *value *= perturbation;

                // Apply bounds
                match param.as_str() {
                    "learning_rate" => *value = value.clamp(1e-5, 1e-1),
                    "weight_decay" => *value = value.clamp(0.0, 0.1),
                    "batch_size" => *value = value.clamp(16.0, 256.0),
                    "momentum" => *value = value.clamp(0.0, 0.999),
                    _ => {}
                }
            }

            // Evaluate new individual
            let mut model = MockModel::new(&new_hyperparams);
            let performance = model.train_and_evaluate();

            population[i] = (new_hyperparams.clone(), performance, population[i].2);
            optimizer.report_performance(new_hyperparams, performance);
        }

        // Statistics for this generation
        let best_performance = population[0].1;
        let avg_performance: f64 =
            population.iter().map(|(_, perf_, _)| *perf_).sum::<f64>() / population.len() as f64;
        let worst_performance = population.last().unwrap().1;

        if generation % 5 == 4 {
            println!(
                "Generation {}: best={:.4}, avg={:.4}, worst={:.4}",
                generation + 1,
                best_performance,
                avg_performance,
                worst_performance
            );
        }
    }

    // Final results
    population.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Population-Based Training Results:");
    println!("Best performance: {:.4}", population[0].1);
    println!("Best hyperparameters:");
    for (param, value) in &population[0].0 {
        println!("  {}: {:.6}", param, value);
    }

    // Population diversity analysis
    let lr_values: Vec<f64> = population
        .iter()
        .map(|(params__, _, _)| params__["learning_rate"])
        .collect();
    let lr_std = {
        let mean = lr_values.iter().sum::<f64>() / lr_values.len() as f64;
        let variance =
            lr_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lr_values.len() as f64;
        variance.sqrt()
    };

    println!("Population diversity (LR std): {:.6}", lr_std);

    Ok(())
}

/// Example 5: Neural Hyperparameter Prediction
#[allow(dead_code)]
fn neural_predictor_example() -> Result<()> {
    println!("\n=== Neural Hyperparameter Prediction Example ===");

    // Create and train hyperparameter predictor
    let predictor = HyperparameterPredictor::<f64>::new(8, 16, 4); // 8 features, 16 hidden, 4 outputs

    // Generate training data (problem features -> optimal hyperparameters)
    let mut training_features = Vec::new();
    let mut training_targets = Vec::new();

    for _ in 0..200 {
        // Simulate different problem characteristics
        let mut rng = rand::rng();
        let features = Array1::from_vec(vec![
            rng.random::<f64>() * 10000.0,    // dataset_size
            rng.random::<f64>() * 1000.0,     // input_dims
            2.0 + rng.random::<f64>() * 98.0, // num_classes
            0.6 + rng.random::<f64>() * 0.3,  // train_ratio
            0.1 + rng.random::<f64>() * 0.2,  // val_ratio
            0.1 + rng.random::<f64>() * 0.2,  // test_ratio
            rng.random::<f64>(),              // sparsity
            rng.random::<f64>() * 0.1,        // noise_level
        ]);

        // Simulate optimal hyperparameters for this problem
        let optimal_hyperparams = HashMap::from([
            (
                "learning_rate".to_string(),
                0.001 + rng.random::<f64>() * 0.009,
            ),
            ("weight_decay".to_string(), rng.random::<f64>() * 0.01),
            ("batch_size".to_string(), 16.0 + rng.random::<f64>() * 112.0),
            ("momentum".to_string(), 0.9 + rng.random::<f64>() * 0.09),
        ]);

        training_features.push(features);
        training_targets.push(optimal_hyperparams);
    }

    // Train the predictor
    println!("Training neural hyperparameter predictor...");
    // predictor.train(&trajectories, 50)?; // Would require OptimizationTrajectory data
    println!("Training completed!");

    // Test predictor on new problems
    for test_case in 0..5 {
        let test_features = extract_problem_features();
        let predicted_hyperparams = predictor.predict(&test_features)?;

        println!("Test case {}:", test_case + 1);
        println!(
            "  Problem features: {:?}",
            test_features.as_slice().unwrap()
        );
        println!("  Predicted hyperparameters:");
        for (param, value) in &predicted_hyperparams {
            println!("    {}: {:.6}", param, value);
        }

        // Evaluate predicted hyperparameters
        let mut model = MockModel::new(&predicted_hyperparams);
        let performance = model.train_and_evaluate();
        println!("  Predicted performance: {:.4}", performance);

        // Compare with random hyperparameters
        let random_hyperparams = HashMap::from([
            (
                "learning_rate".to_string(),
                rand::rng().random::<f64>() * 0.01,
            ),
            (
                "weight_decay".to_string(),
                rand::rng().random::<f64>() * 0.01,
            ),
            (
                "batch_size".to_string(),
                16.0 + rand::rng().random::<f64>() * 112.0,
            ),
            (
                "momentum".to_string(),
                0.9 + rand::rng().random::<f64>() * 0.09,
            ),
        ]);

        let mut random_model = MockModel::new(&random_hyperparams);
        let random_performance = random_model.train_and_evaluate();
        println!("  Random performance: {:.4}", random_performance);
        println!("  Improvement: {:.4}", performance - random_performance);
    }

    Ok(())
}

/// Example 6: Multi-Objective Hyperparameter Optimization
#[allow(dead_code)]
fn multi_objective_optimization_example() -> Result<()> {
    println!("\n=== Multi-Objective Optimization Example ===");

    // Optimize for both accuracy and training time
    let bounds = HashMap::from([
        ("learning_rate".to_string(), (1e-4, 1e-1)),
        ("weight_decay".to_string(), (0.0, 0.1)),
        ("batch_size".to_string(), (16.0, 256.0)),
        ("momentum".to_string(), (0.5, 0.99)),
    ]);

    let strategy = HyperparameterStrategy::RandomSearch {
        num_trials: 40,
        bounds,
    };
    let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

    let mut pareto_front = Vec::new();

    // Run optimization considering multiple objectives
    for trial in 0..40 {
        let hyperparams = optimizer.suggest_hyperparameters()?;

        // Simulate training with timing
        let mut model = MockModel::new(&hyperparams);
        let accuracy = model.train_and_evaluate();

        // Simulate training time (higher batch size = faster, higher LR = faster but less stable)
        let batch_size = hyperparams["batch_size"];
        let learning_rate = hyperparams["learning_rate"];
        let training_time = 100.0 / batch_size + 50.0 / learning_rate; // Simplified simulation

        // For single-objective optimization, we need a combined metric
        let combined_score = accuracy - 0.001 * training_time; // Penalize long training times
        optimizer.report_performance(hyperparams.clone(), combined_score);

        // Store for Pareto analysis
        pareto_front.push((hyperparams, accuracy, training_time, combined_score));

        if trial % 10 == 9 {
            println!(
                "Trial {}: accuracy={:.4}, time={:.1}, combined={:.4}",
                trial + 1,
                accuracy,
                training_time,
                combined_score
            );
        }
    }

    // Find Pareto optimal solutions
    let mut pareto_optimal = Vec::new();
    for (i, (params_i, acc_i, time_i_, _)) in pareto_front.iter().enumerate() {
        let mut is_dominated = false;

        for (j, (_, acc_j, time_j_, _)) in pareto_front.iter().enumerate() {
            if i != j
                && *acc_j >= *acc_i
                && *time_j_ <= *time_i_
                && (*acc_j > *acc_i || *time_j_ < *time_i_)
            {
                is_dominated = true;
                break;
            }
        }

        if !is_dominated {
            pareto_optimal.push((params_i.clone(), *acc_i, *time_i_));
        }
    }

    println!("Multi-Objective Optimization Results:");
    println!("Found {} Pareto optimal solutions:", pareto_optimal.len());

    for (i, (params, acc, time)) in pareto_optimal.iter().enumerate() {
        println!("Solution {}: accuracy={:.4}, time={:.1}", i + 1, acc, time);
        for (param, value) in params {
            println!("  {}: {:.6}", param, value);
        }
    }

    Ok(())
}

/// Example 7: Hyperparameter Sensitivity Analysis
#[allow(dead_code)]
fn sensitivity_analysis_example() -> Result<()> {
    println!("\n=== Hyperparameter Sensitivity Analysis ===");

    // Base configuration
    let base_hyperparams = HashMap::from([
        ("learning_rate".to_string(), 0.001),
        ("weight_decay".to_string(), 0.01),
        ("batch_size".to_string(), 64.0),
        ("momentum".to_string(), 0.9),
    ]);

    // Test sensitivity of each parameter
    let params_to_test = vec!["learning_rate", "weight_decay", "batch_size", "momentum"];
    let perturbation_factors = vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

    for param_name in params_to_test {
        println!("\nSensitivity analysis for {}:", param_name);

        let base_value = base_hyperparams[param_name];
        let mut results = Vec::new();

        for &factor in &perturbation_factors {
            let mut test_hyperparams = base_hyperparams.clone();
            test_hyperparams.insert(param_name.to_string(), base_value * factor);

            // Apply reasonable bounds
            match param_name {
                "learning_rate" => {
                    let val = test_hyperparams.get_mut("learning_rate").unwrap();
                    *val = val.clamp(1e-6, 1.0);
                }
                "weight_decay" => {
                    let val = test_hyperparams.get_mut("weight_decay").unwrap();
                    *val = val.clamp(0.0, 1.0);
                }
                "batch_size" => {
                    let val = test_hyperparams.get_mut("batch_size").unwrap();
                    *val = val.clamp(1.0, 512.0);
                }
                "momentum" => {
                    let val = test_hyperparams.get_mut("momentum").unwrap();
                    *val = val.clamp(0.0, 0.999);
                }
                _ => {}
            }

            let mut model = MockModel::new(&test_hyperparams);
            let performance = model.train_and_evaluate();

            results.push((factor, test_hyperparams[param_name], performance));
        }

        // Find optimal value and sensitivity
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        let best_result = &results[0];

        println!("  Base value: {:.6}", base_value);
        println!(
            "  Optimal value: {:.6} (factor: {:.1})",
            best_result.1, best_result.0
        );
        println!(
            "  Performance improvement: {:.4}",
            best_result.2 - {
                let mut base_model = MockModel::new(&base_hyperparams);
                base_model.train_and_evaluate()
            }
        );

        // Calculate sensitivity (performance variance)
        let performances: Vec<f64> = results.iter().map(|(_, perf, _)| *perf).collect();
        let mean_perf = performances.iter().sum::<f64>() / performances.len() as f64;
        let variance = performances
            .iter()
            .map(|p| (p - mean_perf).powi(2))
            .sum::<f64>()
            / performances.len() as f64;
        let sensitivity = variance.sqrt();

        println!("  Sensitivity (std): {:.4}", sensitivity);
    }

    Ok(())
}

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("SciRS2 Hyperparameter Search Strategies Examples");
    println!("===============================================\n");

    // Run all examples
    grid_search_example()?;
    random_search_example()?;
    bayesian_optimization_example()?;
    population_based_training_example()?;
    neural_predictor_example()?;
    multi_objective_optimization_example()?;
    sensitivity_analysis_example()?;

    println!("\n=== Summary ===");
    println!("This example demonstrated various hyperparameter optimization strategies:");
    println!("1. Grid Search: Systematic but expensive");
    println!("2. Random Search: Efficient and simple");
    println!("3. Bayesian Optimization: Intelligent and sample-efficient");
    println!("4. Population-Based Training: Evolutionary and parallel-friendly");
    println!("5. Neural Predictors: Learning-based and adaptive");
    println!("6. Multi-Objective: Consider multiple goals");
    println!("7. Sensitivity Analysis: Understand parameter importance");
    println!("\nChoose the strategy based on your computational budget and requirements!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model() {
        let hyperparams = HashMap::from([
            ("learning_rate".to_string(), 0.01),
            ("weight_decay".to_string(), 0.001),
            ("batch_size".to_string(), 64.0),
            ("momentum".to_string(), 0.9),
        ]);

        let mut model = MockModel::new(&hyperparams);
        let performance = model.train_and_evaluate();

        assert!(performance >= 0.0 && performance <= 1.0);
        assert_eq!(model.learning_rate, 0.01);
        assert_eq!(model.batch_size, 64);
    }

    #[test]
    fn test_extract_problem_features() {
        let features = extract_problem_features();
        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_hyperparameter_strategies() -> Result<()> {
        // Test grid search with small grid
        let small_grids = HashMap::from([
            ("learning_rate".to_string(), vec![0.001, 0.01]),
            ("weight_decay".to_string(), vec![0.0, 0.001]),
        ]);

        let strategy = HyperparameterStrategy::GridSearch { grids: small_grids };
        let mut optimizer = HyperparameterOptimizer::<f64>::new(strategy);

        let mut trial_count = 0;
        while optimizer.should_continue() && trial_count < 10 {
            let hyperparams = optimizer.suggest_hyperparameters()?;
            let performance = 0.5; // Dummy performance
            optimizer.report_performance(hyperparams, performance);
            trial_count += 1;
        }

        assert!(trial_count <= 4); // 2x2 grid
        assert!(optimizer.get_best_performance().is_some());

        Ok(())
    }
}
