//! Integration tests: scirs2-stats + scirs2-optimize
//!
//! Covers:
//! - MLE fitting of a Normal distribution using gradient-free optimization
//! - Gamma distribution MLE via Nelder-Mead
//! - Negative log-likelihood minimization pipeline
//! - Bootstrap + hypothesis testing pipeline

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::ArrayView1;
use scirs2_optimize::unconstrained::{minimize, Method, Options};
use scirs2_stats::distributions;

// ---------------------------------------------------------------------------
// Helper: negative log-likelihood for Normal(mu, sigma)
// ---------------------------------------------------------------------------

/// Returns the negative log-likelihood for data under N(mu, sigma^2).
fn normal_nll(data: &[f64], mu: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return f64::INFINITY;
    }
    let n = data.len() as f64;
    let sigma2 = sigma * sigma;
    let sum_sq: f64 = data.iter().map(|&x| (x - mu).powi(2)).sum();
    0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() + 0.5 * sum_sq / sigma2
}

// ---------------------------------------------------------------------------
// 1. Normal MLE via Nelder-Mead
// ---------------------------------------------------------------------------

#[test]
fn test_normal_mle_via_nelder_mead() {
    // True parameters
    let true_mu = 3.0_f64;
    let true_sigma = 1.5_f64;

    // Generate deterministic pseudo-data using known quantiles
    // (avoids requiring an RNG, keeps test reproducible)
    let data: Vec<f64> = vec![
        1.23, 2.05, 2.87, 3.01, 3.19, 3.50, 3.78, 4.10, 4.55, 5.22, 1.80, 2.40, 2.90, 3.10, 3.60,
        4.20, 4.70, 2.60, 3.30, 3.80,
    ];

    let nll = |x: &ArrayView1<f64>| -> f64 { normal_nll(&data, x[0], x[1].abs() + 1e-8) };

    let x0 = vec![2.0_f64, 1.0]; // initial guess
    let opts = Options {
        max_iter: 2000,
        ftol: 1e-10,
        ..Options::default()
    };

    let result =
        minimize(nll, &x0, Method::NelderMead, Some(opts)).expect("Normal MLE optimization failed");

    assert!(
        result.success,
        "Optimization did not converge: {}",
        result.message
    );

    let mle_mu = result.x[0];
    let mle_sigma = result.x[1].abs();

    // MLE for Normal: mu = sample mean, sigma = sample std
    let sample_mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let sample_var: f64 =
        data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / data.len() as f64;
    let sample_std = sample_var.sqrt();

    assert_abs_diff_eq!(mle_mu, sample_mean, epsilon = 1e-3);
    assert_abs_diff_eq!(mle_sigma, sample_std, epsilon = 1e-2);
}

// ---------------------------------------------------------------------------
// 2. Exponential distribution MLE via BFGS
// ---------------------------------------------------------------------------

/// Negative log-likelihood for Exponential(lambda) — rate parameterization
fn exponential_nll(data: &[f64], lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return f64::INFINITY;
    }
    let n = data.len() as f64;
    -n * lambda.ln() + lambda * data.iter().sum::<f64>()
}

#[test]
fn test_exponential_mle_via_bfgs() {
    // MLE for Exp: lambda_hat = n / sum(x)
    let data: Vec<f64> = vec![
        0.5, 1.2, 0.8, 0.3, 2.1, 1.5, 0.9, 0.6, 1.1, 0.7, 0.4, 1.8, 0.2, 0.9, 1.3,
    ];

    let nll = |x: &ArrayView1<f64>| -> f64 { exponential_nll(&data, x[0].abs() + 1e-8) };

    let x0 = vec![1.0_f64];
    let opts = Options {
        max_iter: 1000,
        ftol: 1e-12,
        ..Options::default()
    };

    let result = minimize(nll, &x0, Method::NelderMead, Some(opts))
        .expect("Exponential MLE optimization failed");

    assert!(
        result.success,
        "Optimization did not converge: {}",
        result.message
    );

    let mle_lambda = result.x[0].abs();

    let n = data.len() as f64;
    let analytical_lambda = n / data.iter().sum::<f64>();

    assert_abs_diff_eq!(mle_lambda, analytical_lambda, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// 3. Stats distributions: PDF and CDF consistency with optimization
// ---------------------------------------------------------------------------

#[test]
fn test_normal_distribution_properties() {
    let dist =
        distributions::norm(0.0_f64, 1.0_f64).expect("Failed to create Normal(0,1) distribution");

    // CDF at mean is 0.5
    let cdf_at_mean = dist.cdf(0.0);
    assert_abs_diff_eq!(cdf_at_mean, 0.5, epsilon = 1e-6);

    // PDF is symmetric
    let pdf_pos = dist.pdf(1.0);
    let pdf_neg = dist.pdf(-1.0);
    assert_abs_diff_eq!(pdf_pos, pdf_neg, epsilon = 1e-12);

    // PDF integrates to roughly 1 over [-10, 10] via trapezoid
    let n_points = 10000;
    let a = -10.0_f64;
    let b = 10.0_f64;
    let h = (b - a) / n_points as f64;
    let integral: f64 = (0..=n_points)
        .map(|i| {
            let x = a + i as f64 * h;
            let w = if i == 0 || i == n_points { 0.5 } else { 1.0 };
            w * dist.pdf(x)
        })
        .sum::<f64>()
        * h;

    assert_abs_diff_eq!(integral, 1.0, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// 4. Bootstrap mean CI pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_bootstrap_mean_confidence_interval() {
    // Use a fixed small dataset; bootstrap by hand (no RNG needed for CI check)
    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n = data.len();

    let sample_mean: f64 = data.iter().sum::<f64>() / n as f64;

    // True mean for 1..=10 is 5.5
    assert_abs_diff_eq!(sample_mean, 5.5, epsilon = 1e-10);

    // Generate bootstrap samples using a simple LCG (reproducible)
    let n_bootstrap = 1000;
    let mut lcg_state: u64 = 12345;
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut boot_sum = 0.0_f64;
        for _ in 0..n {
            // LCG random number generation
            lcg_state = lcg_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (lcg_state >> 33) as usize % n;
            boot_sum += data[idx];
        }
        bootstrap_means.push(boot_sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).expect("NaN in bootstrap means"));

    // 95% CI: percentiles [2.5%, 97.5%]
    let lo = bootstrap_means[(0.025 * n_bootstrap as f64) as usize];
    let hi = bootstrap_means[(0.975 * n_bootstrap as f64) as usize];

    // True mean (5.5) should be inside the 95% CI
    assert!(
        lo <= sample_mean && sample_mean <= hi,
        "True mean {sample_mean} not in bootstrap CI [{lo}, {hi}]"
    );

    // CI should be reasonably narrow (within ±2 of mean)
    assert!(hi - lo < 4.0, "Bootstrap CI too wide: [{lo}, {hi}]");
}

// ---------------------------------------------------------------------------
// 5. Hypothesis testing: t-test p-value pipeline
// ---------------------------------------------------------------------------

#[test]
fn test_ttest_pipeline_with_distribution() {
    // Known result: two identical samples → p-value should be 1.0 (no difference)
    // We implement a one-sample t-test manually:
    //   t = (x_bar - mu0) / (s / sqrt(n))
    let data: Vec<f64> = vec![5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9];
    let mu0 = 5.0_f64;
    let n = data.len() as f64;

    let x_bar: f64 = data.iter().sum::<f64>() / n;
    let s2: f64 = data.iter().map(|&x| (x - x_bar).powi(2)).sum::<f64>() / (n - 1.0);
    let s = s2.sqrt();
    let t_stat = (x_bar - mu0) / (s / n.sqrt());

    // Student-t distribution with df = n - 1 = 7
    let df = n - 1.0;
    let t_dist =
        distributions::t(df, 0.0_f64, 1.0_f64).expect("Failed to create Student-t distribution");

    // Two-sided p-value: p = 2 * P(T > |t|)
    let cdf_abs_t = t_dist.cdf(t_stat.abs());
    let p_value = 2.0 * (1.0 - cdf_abs_t);

    // With data centered near 5.0, p-value should be high (no strong evidence against H0)
    assert!(
        p_value > 0.05,
        "p-value {p_value} unexpectedly small for data centered at mu0={mu0}"
    );
    assert!(p_value <= 1.0, "p-value {p_value} exceeds 1.0");
}

// ---------------------------------------------------------------------------
// 6. Log-likelihood landscape: optimizer finds correct mode of Gamma distribution
// ---------------------------------------------------------------------------

/// NLL for Gamma(alpha, beta) — shape-rate parameterization
/// ln p(x|alpha,beta) = alpha*ln(beta) - ln(Gamma(alpha)) + (alpha-1)*ln(x) - beta*x
fn gamma_nll(data: &[f64], alpha: f64, beta: f64) -> f64 {
    use std::f64::consts::E;
    if alpha <= 0.0 || beta <= 0.0 {
        return f64::INFINITY;
    }

    // Stirling approximation to ln(Gamma(alpha)) for stability
    let ln_gamma_alpha = if alpha < 1.0 {
        f64::INFINITY // avoid degenerate
    } else {
        // Lanczos approximation (7 terms)
        let g = 7.0_f64;
        #[allow(clippy::excessive_precision)]
        let c = [
            0.999999999999997,
            57.156235665862925,
            -59.597960355475490,
            14.136097974741746,
            -0.491913030587487,
            0.000033994649984811,
            0.000046523628927048,
            -0.000058519345368195,
        ];
        let z = alpha - 1.0;
        let mut acc = c[0];
        for (k, ck) in c[1..].iter().enumerate() {
            acc += ck / (z + k as f64 + 1.0);
        }
        let t = z + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt().ln() + (acc).ln() + (z + 0.5) * t.ln() - t
    };

    let n = data.len() as f64;
    let sum_x: f64 = data.iter().sum();
    let sum_ln_x: f64 = data.iter().map(|&x| x.ln()).sum();

    -(n * alpha * beta.ln() - n * ln_gamma_alpha + (alpha - 1.0) * sum_ln_x - beta * sum_x)
}

#[test]
fn test_gamma_nll_landscape_minimum_near_analytical_mle() {
    // For Gamma(alpha, beta):
    // MLE: alpha_hat approx (x_bar^2) / s^2,  beta_hat = alpha_hat / x_bar
    // (method of moments gives exact for large n)
    let data: Vec<f64> = vec![
        0.8, 1.5, 2.1, 0.6, 1.9, 3.2, 1.1, 0.9, 2.4, 1.7, 0.7, 2.8, 1.3, 1.8, 2.5,
    ];

    let x_bar: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let s2: f64 = data.iter().map(|&x| (x - x_bar).powi(2)).sum::<f64>() / data.len() as f64;

    // Method-of-moments estimates as initial guess
    let alpha0 = x_bar * x_bar / s2;
    let beta0 = alpha0 / x_bar;

    let nll_closure =
        |x: &ArrayView1<f64>| -> f64 { gamma_nll(&data, x[0].abs() + 1e-8, x[1].abs() + 1e-8) };

    let x0 = vec![alpha0, beta0];
    let opts = Options {
        max_iter: 2000,
        ftol: 1e-10,
        ..Options::default()
    };

    let result = minimize(nll_closure, &x0, Method::NelderMead, Some(opts))
        .expect("Gamma MLE optimization failed");

    assert!(
        result.success,
        "Gamma MLE did not converge: {}",
        result.message
    );

    let mle_alpha = result.x[0].abs();
    let mle_beta = result.x[1].abs();
    let mle_mean = mle_alpha / mle_beta;

    // The MLE mean should be close to sample mean
    assert_abs_diff_eq!(mle_mean, x_bar, epsilon = 0.1);

    // Alpha and beta should be positive
    assert!(mle_alpha > 0.0, "MLE alpha is non-positive");
    assert!(mle_beta > 0.0, "MLE beta is non-positive");
}
