//! Advanced statistical analysis for model evaluation
//!
//! This module provides sophisticated statistical analysis tools for model evaluation,
//! including Bayesian model comparison, effect size calculations, and advanced
//! hypothesis testing techniques.

use scirs2_core::ndarray::ArrayStatCompat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

use crate::error::{MetricsError, Result};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;

/// Advanced statistical analysis results
#[derive(Debug, Clone)]
pub struct AdvancedStatisticalResults<F: Float> {
    /// Effect sizes for different metrics
    pub effect_sizes: HashMap<String, F>,
    /// Confidence intervals for metrics
    pub confidence_intervals: HashMap<String, (F, F)>,
    /// Statistical significance tests
    pub significance_tests: HashMap<String, StatisticalTest<F>>,
    /// Bayesian analysis results
    pub bayesian_results: Option<BayesianAnalysisResults<F>>,
    /// Power analysis results
    pub power_analysis: Option<PowerAnalysisResults<F>>,
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest<F: Float> {
    /// Test statistic value
    pub statistic: F,
    /// P-value
    pub p_value: F,
    /// Degrees of freedom (if applicable)
    pub degrees_of_freedom: Option<usize>,
    /// Test name
    pub test_name: String,
    /// Effect size
    pub effect_size: Option<F>,
}

/// Bayesian analysis results
#[derive(Debug, Clone)]
pub struct BayesianAnalysisResults<F: Float> {
    /// Bayes factor comparing models
    pub bayes_factor: F,
    /// Posterior probability of model A being better
    pub posterior_prob_a_better: F,
    /// Credible interval for difference
    pub credible_interval: (F, F),
    /// Expected effect size
    pub expected_effect_size: F,
}

/// Power analysis results
#[derive(Debug, Clone)]
pub struct PowerAnalysisResults<F: Float> {
    /// Statistical power
    pub power: F,
    /// Required sample size for desired power
    pub required_sample_size: usize,
    /// Minimum detectable effect size
    pub min_detectable_effect: F,
    /// Alpha level used
    pub alpha: F,
}

/// Effect size magnitude interpretation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EffectSizeMagnitude {
    Negligible,
    Small,
    Medium,
    Large,
    VeryLarge,
}

/// Advanced statistical analyzer
pub struct AdvancedStatisticalAnalyzer<F: Float> {
    alpha: F,
    beta: F,
    confidence_level: F,
    use_bayesian: bool,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum> Default
    for AdvancedStatisticalAnalyzer<F>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum>
    AdvancedStatisticalAnalyzer<F>
{
    /// Create a new advanced statistical analyzer
    pub fn new() -> Self {
        Self {
            alpha: F::from(0.05).unwrap(),
            beta: F::from(0.20).unwrap(),
            confidence_level: F::from(0.95).unwrap(),
            use_bayesian: true,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set significance level (alpha)
    pub fn with_alpha(mut self, alpha: F) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set Type II error rate (beta)
    pub fn with_beta(mut self, beta: F) -> Self {
        self.beta = beta;
        self
    }

    /// Set confidence level for intervals
    pub fn with_confidence_level(mut self, level: F) -> Self {
        self.confidence_level = level;
        self
    }

    /// Enable/disable Bayesian analysis
    pub fn with_bayesian_analysis(mut self, enable: bool) -> Self {
        self.use_bayesian = enable;
        self
    }

    /// Compare two models using advanced statistical methods
    pub fn compare_models(
        &self,
        model_a_metrics: &HashMap<String, Array1<F>>,
        model_b_metrics: &HashMap<String, Array1<F>>,
    ) -> Result<AdvancedStatisticalResults<F>> {
        let mut results = AdvancedStatisticalResults {
            effect_sizes: HashMap::new(),
            confidence_intervals: HashMap::new(),
            significance_tests: HashMap::new(),
            bayesian_results: None,
            power_analysis: None,
        };

        // Perform analysis for each common metric
        for (metric_name, values_a) in model_a_metrics {
            if let Some(values_b) = model_b_metrics.get(metric_name) {
                // Ensure equal sample sizes
                if values_a.len() != values_b.len() {
                    return Err(MetricsError::InvalidInput(
                        "Sample sizes must be equal for comparison".to_string(),
                    ));
                }

                // Calculate effect size (Cohen's d)
                let effect_size = self.cohensd(values_a.view(), values_b.view())?;
                results
                    .effect_sizes
                    .insert(metric_name.clone(), effect_size);

                // Calculate confidence interval for difference
                let ci = self.confidence_interval_difference(values_a.view(), values_b.view())?;
                results.confidence_intervals.insert(metric_name.clone(), ci);

                // Perform paired t-test
                let t_test = self.paired_t_test(values_a.view(), values_b.view())?;
                results
                    .significance_tests
                    .insert(metric_name.clone(), t_test);

                // Perform Mann-Whitney U test (non-parametric alternative)
                let mann_whitney = self.mann_whitney_u_test(values_a.view(), values_b.view())?;
                results
                    .significance_tests
                    .insert(format!("{}_mann_whitney", metric_name), mann_whitney);
            }
        }

        // Perform Bayesian analysis if enabled
        if self.use_bayesian && !model_a_metrics.is_empty() {
            // Use the first metric for Bayesian analysis
            if let Some((metric_name, values_a)) = model_a_metrics.iter().next() {
                if let Some(values_b) = model_b_metrics.get(metric_name) {
                    let bayesian_results =
                        self.bayesian_model_comparison(values_a.view(), values_b.view())?;
                    results.bayesian_results = Some(bayesian_results);
                }
            }
        }

        // Perform power analysis
        if let Some((_, values_a)) = model_a_metrics.iter().next() {
            if let Some(values_b) = model_b_metrics.values().next() {
                let power_results = self.power_analysis(values_a.view(), values_b.view())?;
                results.power_analysis = Some(power_results);
            }
        }

        Ok(results)
    }

    /// Calculate Cohen's d effect size
    fn cohensd(&self, group_a: ArrayView1<F>, group_b: ArrayView1<F>) -> Result<F> {
        let mean_a = group_a.mean_or(F::zero());
        let mean_b = group_b.mean_or(F::zero());

        let var_a = self.variance(group_a)?;
        let var_b = self.variance(group_b)?;

        let n_a = F::from(group_a.len()).unwrap();
        let n_b = F::from(group_b.len()).unwrap();

        // Pooled standard deviation
        let pooled_sd = ((var_a * (n_a - F::one()) + var_b * (n_b - F::one()))
            / (n_a + n_b - F::from(2).unwrap()))
        .sqrt();

        if pooled_sd == F::zero() {
            return Ok(F::zero());
        }

        Ok((mean_a - mean_b) / pooled_sd)
    }

    /// Calculate confidence interval for difference in means
    fn confidence_interval_difference(
        &self,
        group_a: ArrayView1<F>,
        group_b: ArrayView1<F>,
    ) -> Result<(F, F)> {
        let mean_a = group_a.mean_or(F::zero());
        let mean_b = group_b.mean_or(F::zero());
        let diff = mean_a - mean_b;

        let var_a = self.variance(group_a)?;
        let var_b = self.variance(group_b)?;

        let n_a = F::from(group_a.len()).unwrap();
        let n_b = F::from(group_b.len()).unwrap();

        // Standard error of difference
        let se_diff = (var_a / n_a + var_b / n_b).sqrt();

        // Critical value (approximation for t-distribution)
        let alpha_half = self.alpha / F::from(2).unwrap();
        let t_critical = self.inverse_t_cdf(
            F::one() - alpha_half,
            (n_a + n_b - F::from(2).unwrap()).to_usize().unwrap_or(100),
        )?;

        let margin = t_critical * se_diff;

        Ok((diff - margin, diff + margin))
    }

    /// Perform paired t-test
    fn paired_t_test(
        &self,
        group_a: ArrayView1<F>,
        group_b: ArrayView1<F>,
    ) -> Result<StatisticalTest<F>> {
        let n = group_a.len();
        if n != group_b.len() {
            return Err(MetricsError::InvalidInput(
                "Groups must have equal size for paired t-test".to_string(),
            ));
        }

        if n < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 pairs for t-test".to_string(),
            ));
        }

        // Calculate differences
        let differences: Vec<F> = group_a
            .iter()
            .zip(group_b.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let diff_array = Array1::from(differences);

        let mean_diff = diff_array.mean_or(F::zero());
        let sd_diff = self.std_dev(diff_array.view())?;

        let n_f = F::from(n).unwrap();
        let df = n - 1;

        // Handle zero or near-zero variance case (all differences are identical or nearly so)
        let epsilon = F::epsilon() * F::from(100).unwrap(); // Numerical tolerance
        if sd_diff < epsilon {
            if mean_diff.abs() < epsilon {
                // No difference at all - p-value = 1.0
                return Ok(StatisticalTest {
                    statistic: F::zero(),
                    p_value: F::one(),
                    degrees_of_freedom: Some(df),
                    test_name: "Paired t-test".to_string(),
                    effect_size: Some(F::zero()),
                });
            } else {
                // Perfect or near-perfect consistency in non-zero difference
                // This is extremely significant - use smallest positive value
                return Ok(StatisticalTest {
                    statistic: if mean_diff > F::zero() {
                        F::infinity()
                    } else {
                        -F::infinity()
                    },
                    p_value: F::from(1e-300).unwrap_or(F::epsilon()), // Very small but positive
                    degrees_of_freedom: Some(df),
                    test_name: "Paired t-test".to_string(),
                    effect_size: Some(if mean_diff > F::zero() {
                        F::infinity()
                    } else {
                        -F::infinity()
                    }),
                });
            }
        }

        let t_stat = mean_diff / (sd_diff / n_f.sqrt());

        // Calculate p-value (two-tailed), ensuring it's at least epsilon
        let mut p_value = F::from(2).unwrap() * (F::one() - self.t_cdf(t_stat.abs(), df)?);

        // Clamp p-value to be at least epsilon (avoid exact zero)
        if p_value < F::epsilon() {
            p_value = F::epsilon();
        }

        Ok(StatisticalTest {
            statistic: t_stat,
            p_value,
            degrees_of_freedom: Some(df),
            test_name: "Paired t-test".to_string(),
            effect_size: Some(mean_diff / sd_diff), // Standardized effect size
        })
    }

    /// Perform Mann-Whitney U test (simplified implementation)
    fn mann_whitney_u_test(
        &self,
        group_a: ArrayView1<F>,
        group_b: ArrayView1<F>,
    ) -> Result<StatisticalTest<F>> {
        let n_a = group_a.len();
        let n_b = group_b.len();

        if n_a == 0 || n_b == 0 {
            return Err(MetricsError::InvalidInput(
                "Both groups must be non-empty".to_string(),
            ));
        }

        // Combine and rank all values
        let mut combined: Vec<(F, usize)> = Vec::with_capacity(n_a + n_b);

        for &val in group_a.iter() {
            combined.push((val, 0)); // 0 for group A
        }
        for &val in group_b.iter() {
            combined.push((val, 1)); // 1 for group B
        }

        // Sort by value
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate ranks (handling ties by averaging)
        let mut ranks = vec![F::zero(); combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            // Find all tied values
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }

            // Average rank for tied values
            let avg_rank = F::from(i + j + 1).unwrap() / F::from(2).unwrap();
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        // Sum ranks for group A
        let rank_sum_a: F = combined
            .iter()
            .zip(ranks.iter())
            .filter(|((_, group), _)| *group == 0)
            .map(|(_, &rank)| rank)
            .sum();

        // Calculate U statistic
        let n_a_f = F::from(n_a).unwrap();
        let n_b_f = F::from(n_b).unwrap();
        let u_a = rank_sum_a - n_a_f * (n_a_f + F::one()) / F::from(2).unwrap();
        let u_b = n_a_f * n_b_f - u_a;
        let u_stat = u_a.min(u_b);

        // Approximate p-value using normal approximation
        let mean_u = n_a_f * n_b_f / F::from(2).unwrap();
        let std_u = (n_a_f * n_b_f * (n_a_f + n_b_f + F::one()) / F::from(12).unwrap()).sqrt();

        let z_stat = (u_stat - mean_u) / std_u;
        let p_value = F::from(2).unwrap() * (F::one() - self.standard_normal_cdf(z_stat.abs())?);

        Ok(StatisticalTest {
            statistic: u_stat,
            p_value,
            degrees_of_freedom: None,
            test_name: "Mann-Whitney U test".to_string(),
            effect_size: Some(u_stat / (n_a_f * n_b_f)), // U / (n1 * n2) as effect size
        })
    }

    /// Perform Bayesian model comparison
    fn bayesian_model_comparison(
        &self,
        group_a: ArrayView1<F>,
        group_b: ArrayView1<F>,
    ) -> Result<BayesianAnalysisResults<F>> {
        // Simplified Bayesian analysis using normal-normal model
        let mean_a = group_a.mean_or(F::zero());
        let mean_b = group_b.mean_or(F::zero());

        let var_a = self.variance(group_a)?;
        let var_b = self.variance(group_b)?;

        let n_a = F::from(group_a.len()).unwrap();
        let n_b = F::from(group_b.len()).unwrap();

        // Posterior parameters for difference
        let posterior_mean = mean_a - mean_b;
        let posterior_var = var_a / n_a + var_b / n_b;
        let posterior_std = posterior_var.sqrt();

        // Bayes factor approximation (simplified)
        let effect_size = posterior_mean / posterior_std;
        let bayes_factor = (-effect_size * effect_size / F::from(2).unwrap()).exp();

        // Posterior probability that A is better than B
        let posterior_prob_a_better = F::one() - self.standard_normal_cdf(-effect_size)?;

        // 95% credible interval
        let z_critical = F::from(1.96).unwrap(); // 95% CI
        let margin = z_critical * posterior_std;
        let credible_interval = (posterior_mean - margin, posterior_mean + margin);

        Ok(BayesianAnalysisResults {
            bayes_factor,
            posterior_prob_a_better,
            credible_interval,
            expected_effect_size: effect_size,
        })
    }

    /// Perform power analysis
    fn power_analysis(
        &self,
        group_a: ArrayView1<F>,
        group_b: ArrayView1<F>,
    ) -> Result<PowerAnalysisResults<F>> {
        let effect_size = self.cohensd(group_a, group_b)?;
        let n = F::from(group_a.len()).unwrap();

        // Calculate actual power
        let delta = effect_size * n.sqrt();
        let power = F::one()
            - self.t_cdf(
                self.inverse_t_cdf(
                    F::one() - self.alpha / F::from(2).unwrap(),
                    (F::from(2).unwrap() * n - F::from(2).unwrap())
                        .to_usize()
                        .unwrap_or(100),
                )? - delta,
                (F::from(2).unwrap() * n - F::from(2).unwrap())
                    .to_usize()
                    .unwrap_or(100),
            )?;

        // Required sample size for 80% power
        let desired_power = F::one() - self.beta;
        let z_alpha =
            self.inverse_standard_normal_cdf(F::one() - self.alpha / F::from(2).unwrap())?;
        let z_beta = self.inverse_standard_normal_cdf(desired_power)?;

        let required_n_per_group = ((z_alpha + z_beta) / effect_size).powi(2) * F::from(2).unwrap();
        let required_sample_size = (required_n_per_group * F::from(2).unwrap())
            .ceil()
            .to_usize()
            .unwrap_or(0);

        // Minimum detectable effect size with current sample size
        let min_detectable_effect = (z_alpha + z_beta) / (n / F::from(2).unwrap()).sqrt();

        Ok(PowerAnalysisResults {
            power,
            required_sample_size,
            min_detectable_effect,
            alpha: self.alpha,
        })
    }

    /// Interpret effect size magnitude
    pub fn interpret_effect_size(&self, cohensd: F) -> EffectSizeMagnitude {
        let abs_d = cohensd.abs();

        if abs_d < F::from(0.2).unwrap() {
            EffectSizeMagnitude::Negligible
        } else if abs_d < F::from(0.5).unwrap() {
            EffectSizeMagnitude::Small
        } else if abs_d < F::from(0.8).unwrap() {
            EffectSizeMagnitude::Medium
        } else if abs_d < F::from(0.9).unwrap() {
            EffectSizeMagnitude::Large
        } else {
            EffectSizeMagnitude::VeryLarge
        }
    }

    // Helper statistical functions

    fn variance(&self, data: ArrayView1<F>) -> Result<F> {
        let n = data.len();
        if n < 2 {
            return Ok(F::zero());
        }

        let mean = data.mean_or(F::zero());
        let sum_sq_diff: F = data.iter().map(|&x| (x - mean) * (x - mean)).sum();

        Ok(sum_sq_diff / F::from(n - 1).unwrap())
    }

    fn std_dev(&self, data: ArrayView1<F>) -> Result<F> {
        Ok(self.variance(data)?.sqrt())
    }

    fn t_cdf(&self, t: F, df: usize) -> Result<F> {
        // Use statrs for proper t-distribution CDF
        let t_f64 = t.to_f64().ok_or_else(|| {
            MetricsError::InvalidInput("Cannot convert t-statistic to f64".to_string())
        })?;

        let t_dist = StudentsT::new(0.0, 1.0, df as f64).map_err(|e| {
            MetricsError::InvalidInput(format!("Failed to create t-distribution: {}", e))
        })?;

        let cdf_value = t_dist.cdf(t_f64);
        F::from(cdf_value).ok_or_else(|| {
            MetricsError::InvalidInput("Cannot convert CDF value back to generic type".to_string())
        })
    }

    fn inverse_t_cdf(&self, p: F, df: usize) -> Result<F> {
        // Use statrs for proper inverse t-distribution (quantile function)
        let p_f64 = p.to_f64().ok_or_else(|| {
            MetricsError::InvalidInput("Cannot convert probability to f64".to_string())
        })?;

        let t_dist = StudentsT::new(0.0, 1.0, df as f64).map_err(|e| {
            MetricsError::InvalidInput(format!("Failed to create t-distribution: {}", e))
        })?;

        let quantile = t_dist.inverse_cdf(p_f64);
        F::from(quantile).ok_or_else(|| {
            MetricsError::InvalidInput(
                "Cannot convert quantile value back to generic type".to_string(),
            )
        })
    }

    fn standard_normal_cdf(&self, z: F) -> Result<F> {
        // Approximation using the error function
        let x = z / F::from(2.0).unwrap().sqrt();
        Ok((F::one() + self.erf(x)) / F::from(2).unwrap())
    }

    fn inverse_standard_normal_cdf(&self, p: F) -> Result<F> {
        // Beasley-Springer-Moro algorithm approximation
        let a = [
            F::from(-3.969_683_028_665_376e1).unwrap(),
            F::from(2.209_460_984_245_205e2).unwrap(),
            F::from(-2.759_285_104_469_687e2).unwrap(),
            F::from(1.383_577_518_672_69e2).unwrap(),
            F::from(-3.066_479_806_614_716e1).unwrap(),
            F::from(2.506_628_277_459_239).unwrap(),
        ];

        let b = [
            F::from(-5.447_609_879_822_406e1).unwrap(),
            F::from(1.615_858_368_580_409e2).unwrap(),
            F::from(-1.556_989_798_598_866e2).unwrap(),
            F::from(6.680_131_188_771_972e1).unwrap(),
            F::from(-1.328_068_155_288_572e1).unwrap(),
        ];

        if p <= F::zero() || p >= F::one() {
            return Err(MetricsError::InvalidInput("p must be in (0,1)".to_string()));
        }

        let y = p - F::from(0.5).unwrap();

        if y.abs() < F::from(0.42).unwrap() {
            let r = y * y;
            let mut num = a[5];
            let mut den = F::one();

            for i in (0..5).rev() {
                num = num * r + a[i];
                den = den * r + b[i];
            }

            Ok(y * num / den)
        } else {
            let r = if y > F::zero() { F::one() - p } else { p };
            let r = (-r.ln()).sqrt();

            // Simplified approximation for tail
            Ok(if y > F::zero() { r } else { -r })
        }
    }

    fn erf(&self, x: F) -> F {
        // Approximation for error function
        let a1 = F::from(0.254829592).unwrap();
        let a2 = F::from(-0.284496736).unwrap();
        let a3 = F::from(1.421413741).unwrap();
        let a4 = F::from(-1.453152027).unwrap();
        let a5 = F::from(1.061405429).unwrap();
        let p = F::from(0.3275911).unwrap();

        let sign = if x >= F::zero() { F::one() } else { -F::one() };
        let x = x.abs();

        let t = F::one() / (F::one() + p * x);
        let y = F::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Multi-dimensional effect size calculation
#[allow(dead_code)]
pub fn multi_dimensional_effect_size<
    F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
>(
    metrics_a: &HashMap<String, Array1<F>>,
    metrics_b: &HashMap<String, Array1<F>>,
) -> Result<F> {
    let mut effect_sizes = Vec::new();
    let analyzer = AdvancedStatisticalAnalyzer::new();

    for (metric_name, values_a) in metrics_a {
        if let Some(values_b) = metrics_b.get(metric_name) {
            let effect_size = analyzer.cohensd(values_a.view(), values_b.view())?;
            effect_sizes.push(effect_size);
        }
    }

    if effect_sizes.is_empty() {
        return Ok(F::zero());
    }

    // Calculate Mahalanobis distance-like measure for multidimensional effect size
    let mean_effect: F =
        effect_sizes.iter().cloned().sum::<F>() / F::from(effect_sizes.len()).unwrap();
    let variance: F = effect_sizes
        .iter()
        .map(|&x| (x - mean_effect) * (x - mean_effect))
        .sum::<F>()
        / F::from(effect_sizes.len()).unwrap();

    if variance == F::zero() {
        Ok(mean_effect.abs())
    } else {
        Ok((mean_effect * mean_effect / variance).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cohens_d_calculation() {
        let analyzer = AdvancedStatisticalAnalyzer::<f64>::new();

        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let effect_size = analyzer.cohensd(group_a.view(), group_b.view()).unwrap();
        // The expected value should be approximately -0.632, not -1.0
        assert!((effect_size - (-0.6324555320336759)).abs() < 1e-10);
    }

    #[test]
    fn test_paired_t_test() {
        let analyzer = AdvancedStatisticalAnalyzer::<f64>::new();

        let group_a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let group_b = array![1.1, 2.1, 3.1, 4.1, 5.1];

        let result = analyzer
            .paired_t_test(group_a.view(), group_b.view())
            .unwrap();
        assert!(result.p_value > 0.0);
        assert!(result.p_value <= 1.0);
    }

    #[test]
    fn test_effect_size_interpretation() {
        let analyzer = AdvancedStatisticalAnalyzer::<f64>::new();

        assert_eq!(
            analyzer.interpret_effect_size(0.1),
            EffectSizeMagnitude::Negligible
        );
        assert_eq!(
            analyzer.interpret_effect_size(0.3),
            EffectSizeMagnitude::Small
        );
        assert_eq!(
            analyzer.interpret_effect_size(0.6),
            EffectSizeMagnitude::Medium
        );
        assert_eq!(
            analyzer.interpret_effect_size(0.9),
            EffectSizeMagnitude::VeryLarge
        );
    }

    #[test]
    fn test_model_comparison() {
        let analyzer = AdvancedStatisticalAnalyzer::<f64>::new();

        let mut model_a = HashMap::new();
        let mut model_b = HashMap::new();

        model_a.insert("accuracy".to_string(), array![0.85, 0.87, 0.86, 0.88, 0.84]);
        model_b.insert("accuracy".to_string(), array![0.82, 0.84, 0.83, 0.85, 0.81]);

        let results = analyzer.compare_models(&model_a, &model_b).unwrap();

        assert!(results.effect_sizes.contains_key("accuracy"));
        assert!(results.confidence_intervals.contains_key("accuracy"));
        assert!(results.significance_tests.contains_key("accuracy"));
    }

    #[test]
    fn test_multi_dimensional_effect_size() {
        let mut metrics_a = HashMap::new();
        let mut metrics_b = HashMap::new();

        metrics_a.insert("accuracy".to_string(), array![0.85, 0.87, 0.86]);
        metrics_a.insert("precision".to_string(), array![0.80, 0.82, 0.81]);

        metrics_b.insert("accuracy".to_string(), array![0.82, 0.84, 0.83]);
        metrics_b.insert("precision".to_string(), array![0.77, 0.79, 0.78]);

        let effect_size = multi_dimensional_effect_size(&metrics_a, &metrics_b).unwrap();
        assert!(effect_size > 0.0);
    }
}
