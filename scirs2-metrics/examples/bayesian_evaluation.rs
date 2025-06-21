//! Example of Bayesian evaluation metrics
//!
//! This example demonstrates how to use Bayesian approaches for model evaluation
//! and comparison, including Bayes factors, information criteria, posterior
//! predictive checks, credible intervals, and Bayesian model averaging.

use ndarray::{Array1, Array2};
use scirs2_metrics::bayesian::*;
use scirs2_metrics::error::Result;

fn main() -> Result<()> {
    println!("Bayesian Evaluation Metrics Example");
    println!("==================================");

    // Example 1: Bayesian Model Comparison with Bayes Factors
    println!("\n1. Bayesian Model Comparison");
    println!("---------------------------");

    bayesian_model_comparison_example()?;

    // Example 2: Bayesian Information Criteria
    println!("\n2. Bayesian Information Criteria");
    println!("-------------------------------");

    bayesian_information_criteria_example()?;

    // Example 3: Posterior Predictive Checks
    println!("\n3. Posterior Predictive Checks");
    println!("-----------------------------");

    posterior_predictive_check_example()?;

    // Example 4: Credible Intervals
    println!("\n4. Credible Intervals");
    println!("-------------------");

    credible_interval_example()?;

    // Example 5: Bayesian Model Averaging
    println!("\n5. Bayesian Model Averaging");
    println!("-------------------------");

    bayesian_model_averaging_example()?;

    // Example 6: Comprehensive Bayesian Workflow
    println!("\n6. Comprehensive Bayesian Evaluation Workflow");
    println!("--------------------------------------------");

    comprehensive_bayesian_workflow()?;

    println!("\nBayesian evaluation metrics example completed successfully!");
    Ok(())
}

/// Example of Bayesian model comparison using Bayes factors
fn bayesian_model_comparison_example() -> Result<()> {
    let comparison = BayesianModelComparison::new()
        .with_evidence_method(EvidenceMethod::HarmonicMean)
        .with_num_samples(1000);

    // Simulate log-likelihood samples for two competing models
    // Model A: Better fit (higher likelihood)
    let log_likelihood_a = Array1::from_vec(vec![
        -1.0, -1.2, -0.8, -1.1, -0.9, -1.3, -1.0, -0.7, -1.2, -1.1,
    ]);

    // Model B: Worse fit (lower likelihood)
    let log_likelihood_b = Array1::from_vec(vec![
        -2.0, -2.2, -1.8, -2.1, -1.9, -2.3, -2.0, -1.7, -2.2, -2.1,
    ]);

    // Optional: Add prior information
    let log_prior_a = Array1::from_vec(vec![
        -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
    ]);
    let log_prior_b = Array1::from_vec(vec![
        -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,
    ]);

    let results = comparison.compare_models(
        &log_likelihood_a,
        &log_likelihood_b,
        Some(&log_prior_a),
        Some(&log_prior_b),
    )?;

    println!("Model Comparison Results:");
    println!("  Bayes Factor (A vs B): {:.4}", results.bayes_factor);
    println!("  Log Bayes Factor: {:.4}", results.log_bayes_factor);
    println!("  Evidence A: {:.4}", results.evidence_a);
    println!("  Evidence B: {:.4}", results.evidence_b);
    println!("  Interpretation: {}", results.interpretation);

    // Test different evidence estimation methods
    println!("\nEvidence Estimation Methods:");
    let methods = vec![
        ("Harmonic Mean", EvidenceMethod::HarmonicMean),
        (
            "Thermodynamic Integration",
            EvidenceMethod::ThermodynamicIntegration,
        ),
        ("Bridge Sampling", EvidenceMethod::BridgeSampling),
        ("Nested Sampling", EvidenceMethod::NestedSampling),
    ];

    for (name, method) in methods {
        let comparison_method = BayesianModelComparison::new().with_evidence_method(method);

        let result =
            comparison_method.compare_models(&log_likelihood_a, &log_likelihood_b, None, None)?;

        println!("  {}: BF = {:.4}", name, result.bayes_factor);
    }

    Ok(())
}

/// Example of Bayesian information criteria calculation
fn bayesian_information_criteria_example() -> Result<()> {
    let bic_calc = BayesianInformationCriteria::new().with_num_samples(1000);

    // Simulate log-likelihood samples: 100 MCMC samples, 50 observations
    let log_likelihood_samples = Array2::from_shape_fn((100, 50), |(i, j)| {
        // Simulate realistic log-likelihoods with some variance
        -1.5 - 0.01 * i as f64 - 0.02 * j as f64 + 0.1 * (i as f64 * 0.1).sin()
    });

    let results = bic_calc.evaluate_model(&log_likelihood_samples, 5, 50)?;

    println!("Bayesian Information Criteria:");
    println!("  BIC: {:.4}", results.bic);
    println!("  WAIC: {:.4}", results.waic);
    println!("  LOO-CV: {:.4}", results.loo_cv);
    println!("  DIC: {:.4}", results.dic);
    println!("  Effective Parameters (p_WAIC): {:.4}", results.p_waic);

    // Compare multiple models
    println!("\nMultiple Model Comparison:");
    let mut model_results = Vec::new();

    for (i, num_params) in [3, 5, 8, 12].iter().enumerate() {
        // Simulate different model complexities
        let complexity_factor = 1.0 + 0.1 * i as f64;
        let model_samples = Array2::from_shape_fn((100, 50), |(sample, obs)| {
            -1.5 * complexity_factor - 0.01 * sample as f64 - 0.02 * obs as f64
        });

        let model_result = bic_calc.evaluate_model(&model_samples, *num_params, 50)?;
        model_results.push((*num_params, model_result));
    }

    // Rank models by WAIC (lower is better)
    model_results.sort_by(|a, b| a.1.waic.partial_cmp(&b.1.waic).unwrap());

    println!("Model Ranking by WAIC:");
    for (rank, (params, result)) in model_results.iter().enumerate() {
        println!(
            "  Rank {}: {} parameters, WAIC = {:.4}",
            rank + 1,
            params,
            result.waic
        );
    }

    Ok(())
}

/// Example of posterior predictive checks
fn posterior_predictive_check_example() -> Result<()> {
    // Create observed data
    let observed_data = Array1::from_vec(vec![
        2.1, 2.3, 1.9, 2.0, 2.2, 1.8, 2.1, 2.0, 1.9, 2.3, 2.2, 2.0, 1.8, 2.1, 2.4, 1.9, 2.0, 2.2,
        2.1, 1.9,
    ]);

    // Simulate posterior predictive samples: 1000 samples, 20 observations
    let posterior_predictive_samples = Array2::from_shape_fn((1000, 20), |(sample, obs)| {
        // Simulate data from fitted model with some variation
        let base_mean = 2.0;
        let sample_variation = 0.1 * ((sample as f64 * 0.01).sin());
        let obs_noise = 0.2 * ((obs as f64 * 0.1 + sample as f64 * 0.001).cos());
        base_mean + sample_variation + obs_noise
    });

    // Test different test statistics
    let test_statistics = vec![
        ("Mean", TestStatisticType::Mean),
        ("Variance", TestStatisticType::Variance),
        ("Minimum", TestStatisticType::Minimum),
        ("Maximum", TestStatisticType::Maximum),
    ];

    for (name, test_stat) in test_statistics {
        let ppc = PosteriorPredictiveCheck::new()
            .with_test_statistic(test_stat)
            .with_num_samples(1000);

        let results = ppc.check_model_adequacy(&observed_data, &posterior_predictive_samples)?;

        println!("Posterior Predictive Check ({}):", name);
        println!("  Bayesian p-value: {:.4}", results.bayesian_p_value);
        println!("  Observed statistic: {:.4}", results.observed_statistic);
        println!(
            "  Predicted statistic (mean ± std): {:.4} ± {:.4}",
            results.predicted_statistic_mean, results.predicted_statistic_std
        );
        println!("  Tail probability: {:.4}", results.tail_probability);
        println!("  Model adequate: {}", results.model_adequate);
        println!();
    }

    Ok(())
}

/// Example of credible interval calculation
fn credible_interval_example() -> Result<()> {
    // Simulate posterior samples for a parameter (e.g., treatment effect)
    let posterior_samples = Array1::from_vec(vec![
        0.12, 0.15, 0.18, 0.22, 0.19, 0.16, 0.14, 0.17, 0.20, 0.13, 0.21, 0.18, 0.16, 0.19, 0.17,
        0.15, 0.20, 0.14, 0.18, 0.16, 0.19, 0.17, 0.21, 0.15, 0.18, 0.16, 0.19, 0.17, 0.20, 0.14,
        0.18, 0.16, 0.19, 0.17, 0.15, 0.20, 0.18, 0.16, 0.19, 0.17, 0.14, 0.21, 0.18, 0.16, 0.19,
        0.17, 0.15, 0.20, 0.18, 0.16,
    ]);

    // Test different credible levels
    let credible_levels = vec![0.90, 0.95, 0.99];

    for &level in &credible_levels {
        let ci_calc = CredibleIntervalCalculator::new()
            .with_credible_level(level)
            .with_null_value(0.0); // Test if effect is significantly different from 0

        let results = ci_calc.calculate_intervals(&posterior_samples)?;

        println!("{}% Credible Interval:", (level * 100.0) as u32);
        println!(
            "  Interval: [{:.4}, {:.4}]",
            results.lower_bound, results.upper_bound
        );
        println!("  Posterior mean: {:.4}", results.posterior_mean);
        println!("  Posterior median: {:.4}", results.posterior_median);
        println!("  Contains null (0.0): {}", results.contains_null);
        println!(
            "  HPD interval: [{:.4}, {:.4}]",
            results.hpd_interval.0, results.hpd_interval.1
        );
        println!();
    }

    Ok(())
}

/// Example of Bayesian model averaging
fn bayesian_model_averaging_example() -> Result<()> {
    // Simulate predictions from 4 different models for 10 test cases
    let predictions = Array2::from_shape_vec(
        (4, 10),
        vec![
            // Model 1: Linear model
            1.2, 2.3, 3.1, 4.2, 5.0, 6.1, 7.0, 8.1, 9.0, 10.2,
            // Model 2: Polynomial model
            1.1, 2.4, 3.2, 4.0, 5.1, 6.0, 7.2, 8.0, 9.1, 10.1, // Model 3: Neural network
            1.3, 2.2, 3.0, 4.3, 4.9, 6.2, 6.9, 8.2, 8.9, 10.3, // Model 4: Random forest
            1.0, 2.5, 3.3, 3.9, 5.2, 5.9, 7.3, 7.9, 9.2, 10.0,
        ],
    )
    .unwrap();

    // Model comparison scores (e.g., WAIC values - lower is better)
    let waic_scores = Array1::from_vec(vec![152.3, 148.7, 151.1, 149.9]);

    // Test different weighting methods
    let weighting_methods = vec![
        (
            "Information Criteria",
            ModelWeightingMethod::InformationCriteria,
        ),
        ("Equal Weights", ModelWeightingMethod::Equal),
        ("Cross Validation", ModelWeightingMethod::CrossValidation),
    ];

    for (name, method) in weighting_methods {
        let bma = BayesianModelAveraging::new().with_weighting_method(method);

        let scores = if matches!(method, ModelWeightingMethod::CrossValidation) {
            // For CV, higher is better (e.g., accuracy scores)
            Array1::from_vec(vec![0.85, 0.89, 0.87, 0.88])
        } else {
            waic_scores.clone()
        };

        let results = bma.average_models(&predictions, &scores)?;

        println!("Bayesian Model Averaging ({}):", name);
        println!("  Model weights: {:?}", results.model_weights.to_vec());
        println!(
            "  Averaged predictions: {:?}",
            results.averaged_prediction.to_vec()
        );
        println!(
            "  Model uncertainty (first 5): {:?}",
            results.model_uncertainty.slice(ndarray::s![0..5]).to_vec()
        );
        println!();
    }

    Ok(())
}

/// Comprehensive Bayesian evaluation workflow
fn comprehensive_bayesian_workflow() -> Result<()> {
    println!("This example demonstrates a complete Bayesian evaluation workflow");
    println!("for comparing two regression models on a synthetic dataset.");

    // Step 1: Set up the data and models
    let n_obs = 100;
    let n_samples = 500;

    // Generate synthetic "true" values
    let x_values: Array1<f64> = Array1::linspace(0.0, 10.0, n_obs);
    let true_y: Array1<f64> = x_values.mapv(|x| 2.0 * x + 1.0 + 0.5 * (x * 0.5).sin());

    // Model 1: Linear model log-likelihoods (better fit for linear trend)
    let model1_loglik = Array2::from_shape_fn((n_samples, n_obs), |(s, i)| {
        let prediction = 2.1 * x_values[i] + 0.9;
        let residual = true_y[i] - prediction;
        let noise = 0.1 * ((s as f64 * 0.01).sin());
        -0.5 * (residual + noise).powi(2) / 0.25 - 0.5 * (2.0 * std::f64::consts::PI * 0.5).ln()
    });

    // Model 2: Polynomial model log-likelihoods (captures non-linearity better)
    let model2_loglik = Array2::from_shape_fn((n_samples, n_obs), |(s, i)| {
        let x = x_values[i];
        let prediction = 2.0 * x + 1.0 + 0.4 * (x * 0.5).sin();
        let residual = true_y[i] - prediction;
        let noise = 0.1 * ((s as f64 * 0.01).cos());
        -0.5 * (residual + noise).powi(2) / 0.2 - 0.5 * (2.0 * std::f64::consts::PI * 0.45).ln()
    });

    // Step 2: Model comparison using Bayes factors
    let model_comparison = BayesianModelComparison::new();
    let model1_loglik_sum = model1_loglik.sum_axis(ndarray::Axis(1));
    let model2_loglik_sum = model2_loglik.sum_axis(ndarray::Axis(1));

    let comparison_result =
        model_comparison.compare_models(&model1_loglik_sum, &model2_loglik_sum, None, None)?;

    println!("Model Comparison:");
    println!(
        "  Bayes Factor (Model 1 vs Model 2): {:.4}",
        comparison_result.bayes_factor
    );
    println!("  {}", comparison_result.interpretation);

    // Step 3: Information criteria evaluation
    let bic_calc = BayesianInformationCriteria::new();

    let model1_info = bic_calc.evaluate_model(&model1_loglik, 2, n_obs)?; // Linear: 2 parameters
    let model2_info = bic_calc.evaluate_model(&model2_loglik, 4, n_obs)?; // Polynomial: 4 parameters

    println!("\nInformation Criteria:");
    println!("  Model 1 (Linear):");
    println!("    WAIC: {:.2}", model1_info.waic);
    println!("    LOO-CV: {:.2}", model1_info.loo_cv);
    println!("  Model 2 (Polynomial):");
    println!("    WAIC: {:.2}", model2_info.waic);
    println!("    LOO-CV: {:.2}", model2_info.loo_cv);

    // Step 4: Posterior predictive checks
    let observed_residuals = Array1::from_shape_fn(n_obs, |i| {
        let linear_pred = 2.1 * x_values[i] + 0.9;
        true_y[i] - linear_pred
    });

    let predicted_residuals = Array2::from_shape_fn((n_samples, n_obs), |(s, i)| {
        0.1 * ((s as f64 * 0.01 + i as f64 * 0.1).sin())
    });

    let ppc = PosteriorPredictiveCheck::new().with_test_statistic(TestStatisticType::Variance);
    let ppc_result = ppc.check_model_adequacy(&observed_residuals, &predicted_residuals)?;

    println!("\nPosterior Predictive Check (Residual Variance):");
    println!("  Bayesian p-value: {:.4}", ppc_result.bayesian_p_value);
    println!("  Model adequate: {}", ppc_result.model_adequate);

    // Step 5: Bayesian model averaging
    let model_predictions = Array2::from_shape_fn((2, n_obs), |(model, i)| {
        if model == 0 {
            2.1 * x_values[i] + 0.9 // Linear model
        } else {
            2.0 * x_values[i] + 1.0 + 0.4 * (x_values[i] * 0.5).sin() // Polynomial model
        }
    });

    let waic_scores = Array1::from_vec(vec![model1_info.waic, model2_info.waic]);

    let bma = BayesianModelAveraging::new()
        .with_weighting_method(ModelWeightingMethod::InformationCriteria);

    let bma_result = bma.average_models(&model_predictions, &waic_scores)?;

    println!("\nBayesian Model Averaging:");
    println!(
        "  Model weights: [Linear: {:.3}, Polynomial: {:.3}]",
        bma_result.model_weights[0], bma_result.model_weights[1]
    );
    println!(
        "  Average model uncertainty: {:.4}",
        bma_result.model_uncertainty.mean().unwrap_or(0.0)
    );

    // Step 6: Summary and recommendations
    println!("\nSummary and Recommendations:");

    let best_model = if model1_info.waic < model2_info.waic {
        "Linear"
    } else {
        "Polynomial"
    };
    let waic_diff = (model1_info.waic - model2_info.waic).abs();

    println!(
        "  Best model by WAIC: {} (difference: {:.2})",
        best_model, waic_diff
    );

    if waic_diff < 2.0 {
        println!("  Models are very similar - consider model averaging");
    } else if waic_diff < 6.0 {
        println!("  Moderate evidence favoring {}", best_model);
    } else {
        println!("  Strong evidence favoring {}", best_model);
    }

    if ppc_result.model_adequate {
        println!("  Model assumptions appear adequate based on posterior predictive check");
    } else {
        println!("  Model assumptions may be violated - consider model improvements");
    }

    Ok(())
}
