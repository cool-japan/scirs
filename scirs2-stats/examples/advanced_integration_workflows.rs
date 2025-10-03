//! Advanced Statistical Analysis Integration Examples
//!
//! This example demonstrates the high-level workflows that integrate multiple
//! advanced statistical methods for comprehensive data analysis.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_stats::{
    BayesianAnalysisWorkflow, DimensionalityAnalysisWorkflow, QMCSequenceType, QMCWorkflow,
    SurvivalAnalysisWorkflow,
};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Advanced Statistical Analysis Integration Examples\n");

    // Example 1: Comprehensive Bayesian Analysis
    println!("📊 Example 1: Comprehensive Bayesian Analysis");
    bayesian_analysis_example()?;

    // Example 2: Dimensionality Reduction Analysis
    println!("\n🔍 Example 2: Dimensionality Reduction Analysis");
    dimensionality_analysis_example()?;

    // Example 3: Quasi-Monte Carlo Integration
    println!("\n🎲 Example 3: Quasi-Monte Carlo Integration");
    qmc_analysis_example()?;

    // Example 4: Comprehensive Survival Analysis
    println!("\n⏱️ Example 4: Comprehensive Survival Analysis");
    survival_analysis_example()?;

    Ok(())
}

#[allow(dead_code)]
fn bayesian_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic regression data
    let n_samples = 100;
    let n_features = 3;

    // Create design matrix
    let mut x = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        x[[i, 0]] = 1.0; // Intercept
        x[[i, 1]] = (i as f64) / (n_samples as f64); // Linear trend
        x[[i, 2]] = ((i as f64) / (n_samples as f64)).powi(2); // Quadratic term
    }

    // True coefficients
    let true_beta = array![2.0, 3.0, -1.5];

    // Generate response with noise
    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let noise = 0.1 * (i as f64 % 7.0 - 3.5); // Simple deterministic "noise"
        y[i] = x.row(i).dot(&true_beta) + noise;
    }

    // Create Bayesian analysis workflow
    let workflow = BayesianAnalysisWorkflow::new()
        .with_mcmc(2000, 500)  // 2000 samples, 500 burn-in
        .with_seed(42);

    // Perform comprehensive Bayesian analysis
    let result = workflow.analyze(x.view(), y.view())?;

    println!("Bayesian Linear Regression Results:");
    println!("  Posterior means: {:?}", result.regression.posterior_mean);
    println!("  True coefficients: {:?}", true_beta);
    println!(
        "  Log marginal likelihood: {:.3}",
        result.model_metrics.log_marginal_likelihood
    );
    println!("  DIC: {:.3}", result.model_metrics.dic);
    println!("  WAIC: {:.3}", result.model_metrics.waic);

    if let Some(ref mcmc_samples) = result.mcmc_samples {
        println!("  MCMC samples shape: {:?}", mcmc_samples.dim());
        println!(
            "  Posterior mean from MCMC: {:?}",
            mcmc_samples.mean_axis(scirs2_core::ndarray::Axis(0))
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn dimensionality_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Generate high-dimensional synthetic data with latent structure
    let n_samples = 200;
    let n_features = 10;

    let mut data = Array2::zeros((n_samples, n_features));

    // Create data with 2 underlying factors
    for i in 0..n_samples {
        let factor1 = (i as f64 / n_samples as f64) * 2.0 - 1.0;
        let factor2 = ((i as f64 / n_samples as f64) * 4.0 * std::f64::consts::PI).sin();

        // First 4 features load on factor 1
        for j in 0..4 {
            let loading = 0.7 + 0.3 * (j as f64 / 4.0);
            let noise = 0.1 * (i as f64 % 5.0 - 2.0);
            data[[i, j]] = loading * factor1 + noise;
        }

        // Next 4 features load on factor 2
        for j in 4..8 {
            let loading = 0.6 + 0.4 * ((j - 4) as f64 / 4.0);
            let noise = 0.1 * ((i + j) as f64 % 3.0 - 1.0);
            data[[i, j]] = loading * factor2 + noise;
        }

        // Last 2 features are noise
        for j in 8..10 {
            data[[i, j]] = 0.2 * (i as f64 % 7.0 - 3.0);
        }
    }

    // Create dimensionality analysis workflow
    let workflow = DimensionalityAnalysisWorkflow::new()
        .with_pca(Some(5), false, 1000)  // Up to 5 PCA components
        .with_factor_analysis(Some(3))   // Try 3 factors
        .with_seed(42);

    // Perform comprehensive dimensionality analysis
    let result = workflow.analyze(data.view())?;

    println!("Dimensionality Analysis Results:");

    if let Some(ref pca) = result.pca {
        println!(
            "  PCA explained variance ratio: {:?}",
            pca.explained_variance_ratio
                .slice(scirs2_core::ndarray::s![..5])
        );
        println!(
            "  Cumulative explained variance: {:.3}",
            pca.explained_variance_ratio
                .slice(scirs2_core::ndarray::s![..3])
                .sum()
        );
    }

    if let Some(ref fa) = result.factor_analysis {
        println!("  Factor analysis converged in {} iterations", fa.n_iter);
        println!("  Log-likelihood: {:.3}", fa.log_likelihood);
        println!(
            "  Explained variance by factors: {:?}",
            fa.explained_variance_ratio
        );
    }

    println!("  Recommendations:");
    println!(
        "    Optimal PCA components: {}",
        result.recommendations.optimal_pca_components
    );
    println!(
        "    Optimal factors: {}",
        result.recommendations.optimal_factors
    );
    println!(
        "    Explained variance ratio: {:.3}",
        result.recommendations.explained_variance_ratio
    );

    println!("  Quality metrics:");
    println!(
        "    KMO measure: {:.3}",
        result.comparison_metrics.kmo_measure
    );
    println!(
        "    Bartlett's test: χ²={:.3}, p={:.3}",
        result.comparison_metrics.bartlett_test.0, result.comparison_metrics.bartlett_test.1
    );

    Ok(())
}

#[allow(dead_code)]
fn qmc_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Compare different QMC sequences for numerical integration
    let dimensions = 3;
    let n_samples = 1000;

    println!(
        "Comparing QMC sequences for {}-dimensional integration:",
        dimensions
    );

    // Test different sequence types
    let sequence_types = [
        QMCSequenceType::Sobol,
        QMCSequenceType::Halton,
        QMCSequenceType::LatinHypercube,
    ];

    for sequence_type in &sequence_types {
        let workflow = QMCWorkflow::new(dimensions, n_samples)
            .with_sequence_type(*sequence_type)
            .with_scrambling(true)
            .with_seed(42);

        let result = workflow.generate()?;

        println!("  {:?} sequence:", result.sequence_type);
        println!(
            "    Star discrepancy: {:.6}",
            result.quality_metrics.star_discrepancy
        );
        println!("    Uniformity: {:.3}", result.quality_metrics.uniformity);
        println!(
            "    Coverage efficiency: {:.3}",
            result.quality_metrics.coverage_efficiency
        );

        // Simple integration test: approximate integral of x₁² + x₂² + x₃² over [0,1]³
        // True value is 1.0
        let mut integral_sum = 0.0;
        for i in 0..n_samples {
            let x1 = result.samples[[i, 0]];
            let x2 = result.samples[[i, 1]];
            let x3 = result.samples[[i, 2]];
            integral_sum += x1 * x1 + x2 * x2 + x3 * x3;
        }
        let integral_estimate = integral_sum / n_samples as f64;
        let integration_error = (integral_estimate - 1.0).abs();

        println!(
            "    Integration test (∫(x₁²+x₂²+x₃²)dx): {:.6} (error: {:.6})",
            integral_estimate, integration_error
        );
    }

    Ok(())
}

#[allow(dead_code)]
fn survival_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic survival data
    let n_patients = 100;

    // Survival times (exponential distribution with rate depending on covariates)
    let mut durations = Array1::zeros(n_patients);
    let mut events = Array1::<bool>::default(n_patients);
    let mut covariates = Array2::zeros((n_patients, 2));

    for i in 0..n_patients {
        // Covariate 1: age (scaled)
        covariates[[i, 0]] = (30.0 + (i as f64 / n_patients as f64) * 40.0) / 70.0; // 30-70 years, scaled

        // Covariate 2: treatment (binary)
        covariates[[i, 1]] = if i % 2 == 0 { 1.0 } else { 0.0 };

        // Generate survival time based on Cox model
        // log(hazard) = 0.5 * age + (-0.7) * treatment
        let log_hazard = 0.5 * covariates[[i, 0]] - 0.7 * covariates[[i, 1]];
        let hazard = log_hazard.exp();

        // Generate survival time from exponential distribution
        let u = (i as f64 + 0.5) / n_patients as f64; // Deterministic "random" numbers
        let survival_time = -u.ln() / hazard;

        // Censoring time (exponential with rate 0.3)
        let u_censor = ((i as f64 * 7.0) % 13.0) / 13.0;
        let censor_time = -u_censor.ln() / 0.3;

        if survival_time < censor_time {
            durations[i] = survival_time;
            events[i] = true; // Event observed
        } else {
            durations[i] = censor_time;
            events[i] = false; // Censored
        }
    }

    // Create survival analysis workflow
    let workflow = SurvivalAnalysisWorkflow::new()
        .with_confidence_level(0.95)
        .with_cox_model(100, 1e-6);

    // Perform comprehensive survival analysis
    let result = workflow.analyze(durations.view(), events.view(), Some(covariates.view()))?;

    println!("Survival Analysis Results:");

    // Kaplan-Meier results
    println!("  Kaplan-Meier Estimator:");
    if let Some(median) = result.kaplan_meier.median_survival_time {
        println!("    Median survival time: {:.3}", median);
    } else {
        println!("    Median survival time: not reached");
    }

    println!("  Summary Statistics:");
    println!(
        "    Event rate: {:.1}%",
        result.summary_stats.event_rate * 100.0
    );
    println!(
        "    Censoring rate: {:.1}%",
        result.summary_stats.censoring_rate * 100.0
    );

    if let Some(q25) = result.summary_stats.q25_survival {
        println!("    25th percentile survival: {:.3}", q25);
    }
    if let Some(q75) = result.summary_stats.q75_survival {
        println!("    75th percentile survival: {:.3}", q75);
    }

    // Cox model results
    if let Some(ref cox) = result.cox_model {
        println!("  Cox Proportional Hazards Model:");
        println!("    Converged in {} iterations", cox.n_iter);
        println!("    Log-likelihood: {:.3}", cox.log_likelihood);
        println!("    Coefficients:");
        println!("      Age coefficient: {:.3}", cox.coefficients[0]);
        println!("      Treatment coefficient: {:.3}", cox.coefficients[1]);
        println!("    Hazard ratios:");
        println!("      Age HR: {:.3}", cox.coefficients[0].exp());
        println!("      Treatment HR: {:.3}", cox.coefficients[1].exp());
    }

    Ok(())
}
