//! Advanced analytics for dataset quality assessment
//!
//! This module provides sophisticated analytics capabilities for evaluating
//! dataset quality, complexity, and characteristics.
//!
//! # Provenance note
//!
//! A near-duplicate of this module (`utils::enhanced_analytics`) previously
//! existed side by side with this one: both defined an `AdvancedDatasetAnalyzer`,
//! `AdvancedQualityMetrics`, `NormalityAssessment` and `CorrelationInsights`,
//! but `enhanced_analytics` was never re-exported from `utils::mod` (only
//! reachable via the fully-qualified `utils::enhanced_analytics::...` path),
//! so nothing in the crate or its dependents could reach it. Its genuinely
//! more sophisticated algorithms (histogram-based Shannon entropy/complexity,
//! Mahalanobis multivariate outlier detection, a real Pearson correlation
//! matrix, mutual-information-based feature interactions, and a
//! missing-value-aware ML quality heuristic) have been merged into this
//! (canonical, publicly re-exported) module, and the duplicate file has been
//! deleted. Its `shapiro_wilk_test`/`anderson_darling_test`/`jarque_bera_test`
//! were *not* carried over as-is: `shapiro_wilk_test` was a skewness/kurtosis
//! heuristic mislabeled as Shapiro-Wilk (the same defect this module's real
//! `shapiro_wilk_w` -- Royston 1995, scipy-verified -- already fixed
//! separately), and `anderson_darling_test` didn't compute anything
//! Anderson-Darling-related at all (it just rescaled the Shapiro-Wilk score).
//! Real replacements are implemented below instead (see the private
//! `AdvancedDatasetAnalyzer::anderson_darling_a2` and
//! `AdvancedDatasetAnalyzer::jarque_bera_test` methods).

use super::Dataset;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::statistics::Statistics;
use std::error::Error;

/// Correlation insights from dataset analysis
#[derive(Debug, Clone)]
pub struct CorrelationInsights {
    /// Feature importance scores, derived from each feature's average
    /// absolute linear correlation with every other feature (in `[0, 1]`;
    /// see [`Self::linear_correlations`]).
    pub feature_importance: Array1<f64>,
    /// Pearson linear-correlation matrix between features (`[-1, 1]`,
    /// symmetric, unit diagonal).
    pub linear_correlations: Array2<f64>,
    /// Mutual-information-based estimate of nonlinear association strength
    /// between features (non-negative, symmetric, self-interaction fixed at
    /// `1.0`). Captures dependencies a purely linear correlation misses.
    pub nonlinear_correlations: Array2<f64>,
    /// Rough causality *hints*: half the absolute linear correlation between
    /// each ordered feature pair. This is explicitly **not** a validated
    /// causality test (no real Granger-style temporal/lag structure is
    /// available for generic tabular features) -- it is a deliberately
    /// damped correlation proxy, useful only as an exploratory signal.
    pub causality_hints: Array2<f64>,
}

/// Normality assessment results
#[derive(Debug, Clone)]
pub struct NormalityAssessment {
    /// Overall normality score: a weighted average of
    /// `0.4 * shapiro_wilk + 0.3 * anderson_darling + 0.3 * jarque_bera`
    /// (each already a `[0, 1]`-ish normality score), clamped to `[0, 1]`.
    pub overall_normality: f64,
    /// Shapiro-Wilk W statistic for each feature (Royston 1995 approximation
    /// of the weight vector; see the private
    /// `AdvancedDatasetAnalyzer::shapiro_wilk_w` method).
    /// Values close to 1 indicate the feature is consistent with a normal
    /// distribution; lower values indicate departure from normality.
    pub shapiro_wilk_scores: Array1<f64>,
    /// Anderson-Darling-based normality score for each feature: the real A²
    /// statistic (Anderson & Darling 1952) transformed as `exp(-A²)`, so
    /// larger (more non-normal) A² decays toward 0 and a perfect fit is 1.
    /// See the private `AdvancedDatasetAnalyzer::anderson_darling_a2` method.
    pub anderson_darling_scores: Array1<f64>,
    /// Jarque-Bera-based normality score for each feature: the real JB
    /// statistic (`n/6 * (skewness² + excess_kurtosis²/4)`), converted to an
    /// exact asymptotic p-value via the chi-squared(df=2) survival function
    /// (`exp(-JB/2)`, the closed form of that CDF's complement). Higher
    /// means more consistent with normality.
    pub jarque_bera_scores: Array1<f64>,
}

/// Advanced quality metrics for a dataset
#[derive(Debug, Clone)]
pub struct AdvancedQualityMetrics {
    /// Dataset complexity score: geometric mean, across features, of each
    /// feature's histogram-based Shannon entropy (normalized to `[0, 1]`).
    pub complexity_score: f64,
    /// Information entropy: mean per-feature entropy minus a pairwise
    /// mutual-information correction (an approximation to the joint
    /// entropy), floored at 0.
    pub entropy: f64,
    /// Outlier detection score: fraction of samples whose Mahalanobis
    /// distance from the feature-wise mean exceeds `mean + 3*std` of the
    /// distance distribution (diagonal covariance approximation).
    pub outlier_score: f64,
    /// Machine learning quality score: weighted combination of sample-size,
    /// dimensionality, completeness (non-NaN/non-infinite fraction) and
    /// variance-distribution factors.
    pub ml_quality_score: f64,
    /// Normality assessment results
    pub normality_assessment: NormalityAssessment,
    /// Correlation insights
    pub correlation_insights: CorrelationInsights,
}

/// Advanced dataset analyzer with configurable options
#[derive(Debug, Clone)]
pub struct AdvancedDatasetAnalyzer {
    gpu_enabled: bool,
    advanced_precision: bool,
    significance_threshold: f64,
}

impl Default for AdvancedDatasetAnalyzer {
    fn default() -> Self {
        Self {
            gpu_enabled: false,
            advanced_precision: false,
            significance_threshold: 0.05,
        }
    }
}

impl AdvancedDatasetAnalyzer {
    /// Create a new analyzer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable GPU acceleration
    pub fn with_gpu(mut self, enabled: bool) -> Self {
        self.gpu_enabled = enabled;
        self
    }

    /// Enable advanced precision calculations
    pub fn with_advanced_precision(mut self, enabled: bool) -> Self {
        self.advanced_precision = enabled;
        self
    }

    /// Set significance threshold for statistical tests
    pub fn with_significance_threshold(mut self, threshold: f64) -> Self {
        self.significance_threshold = threshold;
        self
    }

    /// Analyze dataset quality with advanced metrics
    pub fn analyze_dataset_quality(
        &self,
        dataset: &Dataset,
    ) -> Result<AdvancedQualityMetrics, Box<dyn Error>> {
        let data = &dataset.data;
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples < 3 || n_features == 0 {
            return Err(
                "Dataset too small for advanced analysis (need >= 3 samples and >= 1 feature)"
                    .into(),
            );
        }

        // Calculate complexity score based on data distribution
        let complexity_score = self.calculate_complexity_score(data)?;

        // Calculate entropy
        let entropy = self.calculate_entropy(data)?;

        // Calculate outlier score
        let outlier_score = self.calculate_outlier_score(data)?;

        // Calculate ML quality score
        let ml_quality_score = self.calculate_ml_quality_score(data)?;

        // Calculate normality assessment
        let normality_assessment = self.calculate_normality_assessment(data)?;

        // Calculate correlation insights
        let correlation_insights = self.calculate_correlation_insights(data)?;

        Ok(AdvancedQualityMetrics {
            complexity_score,
            entropy,
            outlier_score,
            ml_quality_score,
            normality_assessment,
            correlation_insights,
        })
    }

    /// Dataset complexity via the geometric mean of each feature's
    /// histogram-based Shannon entropy (see [`Self::feature_shannon_entropy`]).
    ///
    /// # History
    ///
    /// This previously ignored the data values entirely: it computed
    /// `(mean_variance.ln() + 1.0).clamp(0.0, 1.0)`, ignoring the actual
    /// *distribution* shape of the data beyond its variance. This real
    /// entropy-based measure was merged in from the (formerly duplicate,
    /// now-deleted) `enhanced_analytics` module.
    fn calculate_complexity_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut product = 1.0_f64;
        for i in 0..n_features {
            product *= self.feature_shannon_entropy(data.column(i))?;
        }
        Ok(product.powf(1.0 / n_features as f64))
    }

    /// Histogram-based Shannon entropy of a single feature, normalized to
    /// `[0, 1]` by the maximum possible entropy for the chosen bin count
    /// (`sqrt(n)` bins, clamped to `[10, 100]`).
    fn feature_shannon_entropy(&self, feature: ArrayView1<f64>) -> Result<f64, Box<dyn Error>> {
        let mut values = feature.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_bins = ((values.len() as f64).sqrt() as usize).clamp(10, 100);
        let min_val = values[0];
        let max_val = values[values.len() - 1];

        if (max_val - min_val).abs() < f64::EPSILON {
            return Ok(0.0); // Constant feature has zero complexity/entropy.
        }

        let bin_width = (max_val - min_val) / n_bins as f64;
        let mut histogram = vec![0usize; n_bins];
        for &value in &values {
            let bin_idx = ((value - min_val) / bin_width) as usize;
            histogram[bin_idx.min(n_bins - 1)] += 1;
        }

        let n_total = values.len() as f64;
        let entropy = histogram
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / n_total;
                -p * p.ln()
            })
            .sum::<f64>();

        let max_entropy = (n_bins as f64).ln();
        Ok(entropy / max_entropy)
    }

    /// Dataset (joint) entropy: mean per-feature Shannon entropy, corrected
    /// downward by an estimate of the pairwise mutual information (shared,
    /// non-independent information should not be double-counted), floored
    /// at 0.
    ///
    /// # History
    ///
    /// This previously computed `(n_samples.ln() / 2.0).clamp(0.0, 5.0)` --
    /// a function of the sample *count* only, completely independent of the
    /// data values. Merged in from `enhanced_analytics`.
    fn calculate_entropy(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        let n_features = data.ncols();

        let mean_entropy = (0..n_features)
            .map(|i| self.feature_shannon_entropy(data.column(i)).unwrap_or(0.0))
            .sum::<f64>()
            / n_features as f64;

        let mutual_info_correction = self.estimate_mutual_information(data)?;

        Ok((mean_entropy * n_features as f64 - mutual_info_correction).max(0.0))
    }

    /// Average pairwise mutual information across a bounded sample of
    /// feature pairs (capped at 100 pairs for efficiency on wide datasets).
    fn estimate_mutual_information(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        let n_features = data.ncols();
        if n_features < 2 {
            return Ok(0.0);
        }

        let max_pairs = 100;
        let total_pairs = n_features * (n_features - 1) / 2;
        let step = (total_pairs / max_pairs).max(1);

        let mut total_mi = 0.0;
        let mut pair_count = 0;
        for i in (0..n_features).step_by(step) {
            for j in (i + 1..n_features).step_by(step) {
                total_mi += self.pairwise_mutual_information(data.column(i), data.column(j))?;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 {
            total_mi / pair_count as f64
        } else {
            0.0
        })
    }

    /// Mutual information between two features via a 2D histogram estimate
    /// (20x20 bins). Non-negative by construction (clamped at 0 to guard
    /// against small negative values from binning noise).
    fn pairwise_mutual_information(
        &self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
    ) -> Result<f64, Box<dyn Error>> {
        let n_bins = 20;

        let x_min = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let x_max = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let y_min = y.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let y_max = y.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (x_max - x_min).abs() < f64::EPSILON || (y_max - y_min).abs() < f64::EPSILON {
            return Ok(0.0);
        }

        let x_bin_width = (x_max - x_min) / n_bins as f64;
        let y_bin_width = (y_max - y_min) / n_bins as f64;

        let mut joint_hist = vec![vec![0usize; n_bins]; n_bins];
        let mut x_hist = vec![0usize; n_bins];
        let mut y_hist = vec![0usize; n_bins];

        let n_samples = x.len();
        for i in 0..n_samples {
            let x_bin = (((x[i] - x_min) / x_bin_width) as usize).min(n_bins - 1);
            let y_bin = (((y[i] - y_min) / y_bin_width) as usize).min(n_bins - 1);
            joint_hist[x_bin][y_bin] += 1;
            x_hist[x_bin] += 1;
            y_hist[y_bin] += 1;
        }

        let n_total = n_samples as f64;
        let mut mi = 0.0;
        for (i, xi) in x_hist.iter().enumerate() {
            for (j, yj) in y_hist.iter().enumerate() {
                let joint = joint_hist[i][j];
                if joint > 0 && *xi > 0 && *yj > 0 {
                    let p_xy = joint as f64 / n_total;
                    let p_x = *xi as f64 / n_total;
                    let p_y = *yj as f64 / n_total;
                    mi += p_xy * (p_xy / (p_x * p_y)).ln();
                }
            }
        }

        Ok(mi.max(0.0))
    }

    /// Multivariate outlier score: fraction of samples whose Mahalanobis
    /// distance from the mean (diagonal-covariance approximation) exceeds
    /// `mean_distance + 3*std_distance`.
    ///
    /// # History
    ///
    /// This previously used a per-feature (univariate) z-score check, which
    /// cannot detect a sample that is unremarkable on every individual
    /// feature but jointly anomalous. Merged in from `enhanced_analytics`
    /// (which already documented the diagonal-covariance approximation as a
    /// deliberate simplification in place of a full matrix inverse).
    fn calculate_outlier_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        let n_samples = data.nrows();
        if n_samples < 3 {
            return Ok(0.0);
        }

        let mean = data.mean_axis(Axis(0)).ok_or("failed to compute mean")?;
        let variances: Vec<f64> = (0..data.ncols()).map(|i| data.column(i).var(1.0)).collect();

        let distances: Vec<f64> = (0..n_samples)
            .map(|k| {
                let mut distance_squared = 0.0;
                for (i, &variance) in variances.iter().enumerate() {
                    if variance > f64::EPSILON {
                        distance_squared += (data[[k, i]] - mean[i]).powi(2) / variance;
                    }
                }
                distance_squared.sqrt()
            })
            .collect();

        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let distance_std = {
            let variance = distances
                .iter()
                .map(|&d| (d - mean_distance).powi(2))
                .sum::<f64>()
                / distances.len() as f64;
            variance.sqrt()
        };

        let threshold = mean_distance + 3.0 * distance_std;
        let outlier_count = distances.iter().filter(|&&d| d > threshold).count();

        Ok(outlier_count as f64 / n_samples as f64)
    }

    /// Machine-learning-oriented quality score combining sample size,
    /// dimensionality, completeness (non-NaN/non-infinite fraction), and how
    /// evenly feature variances are distributed.
    ///
    /// # History
    ///
    /// This previously only looked at `(mean_variance.ln() + 5.0) / 10.0`,
    /// with no notion of dataset size or missing/non-finite values at all
    /// (a dataset riddled with `NaN` would silently poison `mean_variance`
    /// with no completeness penalty applied). Merged in from
    /// `enhanced_analytics`.
    fn calculate_ml_quality_score(&self, data: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        if n_samples < 10 || n_features == 0 {
            return Ok(0.1); // Low confidence for very small datasets.
        }

        let size_factor = (n_samples as f64 / (n_samples as f64 + 100.0)).min(1.0);
        let dimensionality_factor = (n_features as f64 / (n_features as f64 + 50.0)).min(1.0);

        let missing_rate = {
            let total = data.len();
            let missing = data.iter().filter(|&&x| !x.is_finite()).count();
            missing as f64 / total as f64
        };
        let completeness_factor = 1.0 - missing_rate;

        let variance_factor = {
            let variances: Vec<f64> = (0..n_features).map(|i| data.column(i).var(1.0)).collect();
            let mean_variance = variances.iter().sum::<f64>() / n_features as f64;

            if mean_variance <= f64::EPSILON {
                0.1 // All-constant features: low quality.
            } else {
                let variance_of_variances = variances
                    .iter()
                    .map(|&v| (v - mean_variance).powi(2))
                    .sum::<f64>()
                    / n_features as f64;
                let variance_cv = variance_of_variances.sqrt() / mean_variance;
                (1.0 / (1.0 + variance_cv)).max(0.1)
            }
        };

        let quality_score = (size_factor * 0.25
            + dimensionality_factor * 0.15
            + completeness_factor * 0.35
            + variance_factor * 0.25)
            .clamp(0.0, 1.0);

        Ok(quality_score)
    }

    fn calculate_normality_assessment(
        &self,
        data: &Array2<f64>,
    ) -> Result<NormalityAssessment, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut shapiro_scores = Vec::with_capacity(n_features);
        let mut anderson_scores = Vec::with_capacity(n_features);
        let mut jarque_scores = Vec::with_capacity(n_features);

        for col in 0..n_features {
            let column = data.column(col);
            shapiro_scores.push(self.shapiro_wilk_w(&column)?);
            anderson_scores.push(self.anderson_darling_score(column)?);
            jarque_scores.push(self.jarque_bera_test(column)?);
        }

        let shapiro_wilk_scores = Array1::from_vec(shapiro_scores);
        let anderson_darling_scores = Array1::from_vec(anderson_scores);
        let jarque_bera_scores = Array1::from_vec(jarque_scores);

        let mean_of = |scores: &Array1<f64>| {
            let val = scores.view().mean();
            if val.is_nan() {
                0.5
            } else {
                val
            }
        };
        let overall_normality = (mean_of(&shapiro_wilk_scores) * 0.4
            + mean_of(&anderson_darling_scores) * 0.3
            + mean_of(&jarque_bera_scores) * 0.3)
            .clamp(0.0, 1.0);

        Ok(NormalityAssessment {
            overall_normality,
            shapiro_wilk_scores,
            anderson_darling_scores,
            jarque_bera_scores,
        })
    }

    /// Computes the Shapiro-Wilk W statistic for a normality test, using the
    /// Royston (1995, AS R94) polynomial approximation for the order-statistic
    /// weight vector -- the same coefficient set SciPy's `scipy.stats.shapiro`
    /// and R's `shapiro.test` use to build their weights.
    ///
    /// Returns a value in `[0, 1]`: values close to 1 indicate the sample is
    /// consistent with a normal distribution, lower values indicate
    /// departure from normality. This computes the real W test statistic
    /// (not merely a skewness/kurtosis heuristic); the p-value transform is
    /// intentionally not computed since callers here only need a bounded
    /// normality *score*, not a hypothesis-test decision.
    ///
    /// Returns `0.5` (a neutral "undetermined") for `n < 3`, where the
    /// statistic isn't defined, and `1.0` for exactly-constant data (zero
    /// variance, where the ratio is otherwise `0/0`).
    fn shapiro_wilk_w(
        &self,
        data: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> Result<f64, Box<dyn Error>> {
        let n = data.len();
        if n < 3 {
            return Ok(0.5);
        }

        let mut x: Vec<f64> = data.iter().copied().collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let xbar = x.iter().sum::<f64>() / n as f64;
        let denom: f64 = x.iter().map(|&xi| (xi - xbar).powi(2)).sum();
        if denom <= f64::EPSILON {
            // Constant data: not meaningfully non-normal in this context.
            return Ok(1.0);
        }

        let normal = Normal::new(0.0, 1.0)?;
        let nf = n as f64;
        let n2 = n / 2;

        // Expected order-statistic means for the lower half of the sample
        // (Blom's approximation), m[i-1] corresponds to the i-th order
        // statistic (1-indexed) for i = 1..=n2.
        let m: Vec<f64> = (1..=n2)
            .map(|i| normal.inverse_cdf((i as f64 - 0.375) / (nf + 0.25)))
            .collect();

        // 1-indexed weight vector; a[0] is unused padding.
        let mut a = vec![0.0_f64; n + 1];

        if n == 3 {
            // Exact analytical weights for n = 3 (the polynomial
            // approximation below is only calibrated for n >= 4).
            a[1] = -std::f64::consts::FRAC_1_SQRT_2;
            a[3] = std::f64::consts::FRAC_1_SQRT_2;
        } else {
            let ssumm2 = 2.0 * m.iter().map(|v| v * v).sum::<f64>();
            let rsn = 1.0 / nf.sqrt();

            // Royston (1995) polynomial coefficients for the two largest
            // weights, evaluated at rsn = 1/sqrt(n) via Horner's method.
            const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056];
            const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];
            let poly = |c: &[f64; 6], t: f64| -> f64 {
                c.iter().rev().fold(0.0, |acc, &coef| acc * t + coef)
            };

            let m_n = -m[0]; // largest order-statistic mean
            let m_n1 = -m[1]; // second largest (n2 >= 2 whenever n >= 4)
            let a_n = m_n / ssumm2.sqrt() + poly(&C1, rsn);
            let a_n1 = m_n1 / ssumm2.sqrt() + poly(&C2, rsn);
            a[n] = a_n;

            let (phi, loop_start) = if n > 5 {
                a[n - 1] = a_n1;
                let phi = (ssumm2 - 2.0 * m_n * m_n - 2.0 * m_n1 * m_n1)
                    / (1.0 - 2.0 * a_n * a_n - 2.0 * a_n1 * a_n1);
                (phi, 3)
            } else {
                let phi = (ssumm2 - 2.0 * m_n * m_n) / (1.0 - 2.0 * a_n * a_n);
                (phi, 2)
            };

            for i in loop_start..=n2 {
                let mi = -m[i - 1];
                a[n + 1 - i] = mi / phi.sqrt();
            }

            // Antisymmetric lower half; a[n2+1] (odd n's middle element)
            // stays at its default 0.0.
            for i in 1..=n2 {
                a[i] = -a[n + 1 - i];
            }
        }

        let numerator = (0..n).map(|i| a[i + 1] * x[i]).sum::<f64>().powi(2);
        let w = numerator / denom;

        Ok(w.clamp(0.0, 1.0))
    }

    /// Anderson-Darling A² statistic (Anderson & Darling 1952) against a
    /// fitted normal distribution:
    ///
    /// ```text
    /// A² = -n - (1/n) * sum_{i=1}^{n} (2i-1) * [ln Φ(z_(i)) + ln(1 - Φ(z_(n+1-i)))]
    /// ```
    ///
    /// where `z_(i)` are the standardized (mean 0, unit variance), ascending
    /// order statistics of the sample and `Φ` is the standard normal CDF.
    /// Larger `A²` indicates greater departure from normality; `0` for
    /// (theoretically) constant data or `n < 2` where the statistic is
    /// undefined/degenerate.
    fn anderson_darling_a2(&self, data: ArrayView1<f64>) -> Result<f64, Box<dyn Error>> {
        let n = data.len();
        if n < 2 {
            return Ok(0.0);
        }

        let mean = {
            let val = data.mean();
            if val.is_nan() {
                0.0
            } else {
                val
            }
        };
        let variance = data.var(1.0);
        if variance <= f64::EPSILON {
            return Ok(0.0); // Constant data: treat as a perfect (trivial) fit.
        }
        let std_dev = variance.sqrt();

        let mut z: Vec<f64> = data.iter().map(|&x| (x - mean) / std_dev).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let normal = Normal::new(0.0, 1.0)?;
        // Clamp away from exactly 0/1 so ln(.) stays finite even for
        // extreme standardized values.
        let eps = 1e-15;
        let phi = |x: f64| normal.cdf(x).clamp(eps, 1.0 - eps);

        let nf = n as f64;
        let mut sum = 0.0;
        for (idx, &zi) in z.iter().enumerate() {
            let i = idx + 1; // 1-indexed order statistic position
            let z_complement = z[n - i]; // z_(n+1-i), 0-indexed as z[n-i]
            let weight = 2.0 * i as f64 - 1.0;
            sum += weight * (phi(zi).ln() + (1.0 - phi(z_complement)).ln());
        }

        let a2 = -nf - sum / nf;
        Ok(a2.max(0.0))
    }

    /// Anderson-Darling-based normality *score* in `[0, 1]`: `exp(-A²)`, so a
    /// perfect fit (A² = 0) scores 1 and departure from normality decays the
    /// score toward 0. See [`Self::anderson_darling_a2`] for the real
    /// underlying statistic.
    fn anderson_darling_score(&self, data: ArrayView1<f64>) -> Result<f64, Box<dyn Error>> {
        let a2 = self.anderson_darling_a2(data)?;
        Ok((-a2).exp().clamp(0.0, 1.0))
    }

    /// Jarque-Bera normality score: the real JB statistic
    /// `n/6 * (skewness² + excess_kurtosis²/4)`, converted to its exact
    /// asymptotic p-value via the chi-squared(df=2) survival function. Since
    /// the CDF of a chi-squared distribution with 2 degrees of freedom has
    /// the closed form `1 - exp(-x/2)`, the survival function (p-value) is
    /// exactly `exp(-JB/2)` -- no numerical integration or table lookup
    /// needed. Higher (closer to 1) means more consistent with normality.
    ///
    /// # History
    ///
    /// The JB statistic itself was already computed correctly by the
    /// duplicate `enhanced_analytics` module this was merged from, but its
    /// final step (`(-jb_stat / 10.0).exp()`) was an arbitrary rescaling
    /// with no statistical meaning; this replaces it with the real
    /// chi-squared(2) p-value.
    fn jarque_bera_test(&self, data: ArrayView1<f64>) -> Result<f64, Box<dyn Error>> {
        let n = data.len();
        if n < 3 {
            return Ok(0.5);
        }

        let mean = {
            let val = data.mean();
            if val.is_nan() {
                0.0
            } else {
                val
            }
        };
        let variance = data.var(1.0);
        if variance <= f64::EPSILON {
            return Ok(1.0);
        }
        let std_dev = variance.sqrt();

        let skewness = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n as f64;
        let excess_kurtosis = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n as f64
            - 3.0;

        let jb_stat = (n as f64 / 6.0) * (skewness.powi(2) + excess_kurtosis.powi(2) / 4.0);

        // Exact chi-squared(df=2) survival function at jb_stat.
        Ok((-jb_stat / 2.0).exp())
    }

    fn calculate_correlation_insights(
        &self,
        data: &Array2<f64>,
    ) -> Result<CorrelationInsights, Box<dyn Error>> {
        let n_features = data.ncols();

        let linear_correlations = self.calculate_correlation_matrix(data)?;
        let nonlinear_correlations = self.calculate_interaction_matrix(data)?;
        let causality_hints = self.estimate_causality_matrix(&linear_correlations);

        // Feature importance: average absolute linear correlation with every
        // other feature (a feature strongly related to the rest of the
        // dataset is, in this sense, more "important").
        //
        // # History
        //
        // This previously ranked importance purely by each feature's *own*
        // variance (`(variance.ln() + 1.0).clamp(0.0, 1.0)`), never actually
        // looking at cross-feature relationships despite living inside
        // `CorrelationInsights`. The correlation-based version below was
        // merged in from `enhanced_analytics`.
        let feature_importance = if n_features > 1 {
            Array1::from_vec(
                (0..n_features)
                    .map(|i| {
                        let total: f64 = (0..n_features)
                            .filter(|&j| j != i)
                            .map(|j| linear_correlations[[i, j]].abs())
                            .sum();
                        total / (n_features - 1) as f64
                    })
                    .collect(),
            )
        } else {
            Array1::from_vec(vec![1.0; n_features])
        };

        Ok(CorrelationInsights {
            feature_importance,
            linear_correlations,
            nonlinear_correlations,
            causality_hints,
        })
    }

    /// Symmetric Pearson linear-correlation matrix (unit diagonal).
    fn calculate_correlation_matrix(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut corr_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let correlation = if i == j {
                    1.0
                } else {
                    self.pearson_correlation(data.column(i), data.column(j))
                };
                corr_matrix[[i, j]] = correlation;
                corr_matrix[[j, i]] = correlation;
            }
        }

        Ok(corr_matrix)
    }

    /// Pearson product-moment correlation coefficient. Returns `0.0` for
    /// mismatched/degenerate inputs (fewer than 2 samples, or zero variance
    /// in either series, where the coefficient is otherwise `0/0`).
    fn pearson_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len();
        if n != y.len() || n < 2 {
            return 0.0;
        }

        let mean_x = {
            let val = x.mean();
            if val.is_nan() {
                0.0
            } else {
                val
            }
        };
        let mean_y = {
            let val = y.mean();
            if val.is_nan() {
                0.0
            } else {
                val
            }
        };

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator <= f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Mutual-information-based feature interaction matrix (unit diagonal
    /// for self-interaction, symmetric, non-negative off-diagonal).
    fn calculate_interaction_matrix(
        &self,
        data: &Array2<f64>,
    ) -> Result<Array2<f64>, Box<dyn Error>> {
        let n_features = data.ncols();
        let mut interaction_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in i..n_features {
                let interaction = if i == j {
                    1.0
                } else {
                    self.pairwise_mutual_information(data.column(i), data.column(j))?
                };
                interaction_matrix[[i, j]] = interaction;
                interaction_matrix[[j, i]] = interaction;
            }
        }

        Ok(interaction_matrix)
    }

    /// Rough causality *hints* from an already-computed correlation matrix:
    /// half the absolute correlation for each ordered pair (see
    /// [`CorrelationInsights::causality_hints`] for why this is explicitly
    /// not a real causality test).
    fn estimate_causality_matrix(&self, linear_correlations: &Array2<f64>) -> Array2<f64> {
        let n_features = linear_correlations.nrows();
        let mut causality_matrix = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                // Diagonal left at 0.0: self-"causality" isn't meaningful.
                // (Not implemented as `correlation == 1.0`, which would also
                // zero out any genuinely perfectly-correlated *off-diagonal*
                // feature pair -- e.g. one feature that is an exact linear
                // function of another.)
                if i != j {
                    causality_matrix[[i, j]] = linear_correlations[[i, j]].abs() * 0.5;
                }
            }
        }
        causality_matrix
    }
}

/// Perform quick quality assessment of a dataset
pub fn quick_quality_assessment(dataset: &Dataset) -> Result<f64, Box<dyn Error>> {
    let data = &dataset.data;

    // Quick quality assessment based on basic statistics
    let n_samples = data.nrows();
    let n_features = data.ncols();

    if n_samples == 0 || n_features == 0 {
        return Ok(0.0);
    }

    // Check for missing values (NaN/inf)
    let valid_count = data.iter().filter(|&&x| x.is_finite()).count();
    let completeness = valid_count as f64 / data.len() as f64;

    // Check feature variance
    let variances: Array1<f64> = data.var_axis(scirs2_core::ndarray::Axis(0), 1.0);
    let non_zero_var_count = variances.iter().filter(|&&x| x > 1e-10).count();
    let variance_score = non_zero_var_count as f64 / n_features as f64;

    // Simple size penalty for very small datasets
    let size_score = ((n_samples as f64).ln() / 10.0).clamp(0.0, 1.0);

    // Combined quality score
    let quality_score = (completeness + variance_score + size_score) / 3.0;

    Ok(quality_score.clamp(0.0, 1.0))
}

/// Advanced dataset analysis function
#[allow(dead_code)]
pub fn analyze_dataset_advanced(
    dataset: &Dataset,
) -> Result<AdvancedQualityMetrics, Box<dyn Error>> {
    let analyzer = AdvancedDatasetAnalyzer::new()
        .with_gpu(false)
        .with_advanced_precision(true)
        .with_significance_threshold(0.05);

    analyzer.analyze_dataset_quality(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_quick_quality_assessment() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect())
            .expect("Operation failed");
        let dataset = Dataset::new(data, None);

        let quality = quick_quality_assessment(&dataset).expect("Operation failed");
        assert!((0.0..=1.0).contains(&quality));
    }

    #[test]
    fn test_advanced_dataset_analyzer() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect())
            .expect("Operation failed");
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new()
            .with_gpu(false)
            .with_advanced_precision(true);

        let metrics = analyzer
            .analyze_dataset_quality(&dataset)
            .expect("Operation failed");
        assert!(metrics.complexity_score >= 0.0);
        assert!(metrics.entropy >= 0.0);
        assert!(metrics.outlier_score >= 0.0);
        assert!(metrics.ml_quality_score >= 0.0);
    }

    #[test]
    fn test_normality_assessment() {
        let data = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as f64).collect())
            .expect("Operation failed");
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new();
        let metrics = analyzer
            .analyze_dataset_quality(&dataset)
            .expect("Operation failed");

        assert!(metrics.normality_assessment.overall_normality >= 0.0);
        assert!(metrics.normality_assessment.overall_normality <= 1.0);
        assert_eq!(metrics.normality_assessment.shapiro_wilk_scores.len(), 2);
        assert_eq!(
            metrics.normality_assessment.anderson_darling_scores.len(),
            2
        );
        assert_eq!(metrics.normality_assessment.jarque_bera_scores.len(), 2);
        assert!(metrics
            .normality_assessment
            .anderson_darling_scores
            .iter()
            .all(|&x| (0.0..=1.0).contains(&x)));
        assert!(metrics
            .normality_assessment
            .jarque_bera_scores
            .iter()
            .all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_correlation_insights() {
        let data = Array2::from_shape_vec((15, 3), (0..45).map(|x| x as f64).collect())
            .expect("Operation failed");
        let dataset = Dataset::new(data, None);

        let analyzer = AdvancedDatasetAnalyzer::new();
        let metrics = analyzer
            .analyze_dataset_quality(&dataset)
            .expect("Operation failed");

        assert_eq!(metrics.correlation_insights.feature_importance.len(), 3);
        assert!(metrics
            .correlation_insights
            .feature_importance
            .iter()
            .all(|&x| (0.0..=1.0).contains(&x)));

        // New (merged-in) correlation-analysis fields.
        assert_eq!(
            metrics.correlation_insights.linear_correlations.shape(),
            &[3, 3]
        );
        assert_eq!(
            metrics.correlation_insights.nonlinear_correlations.shape(),
            &[3, 3]
        );
        assert_eq!(
            metrics.correlation_insights.causality_hints.shape(),
            &[3, 3]
        );
        // Linear correlation matrix must be symmetric with a unit diagonal.
        for i in 0..3 {
            assert!((metrics.correlation_insights.linear_correlations[[i, i]] - 1.0).abs() < 1e-9);
            for j in 0..3 {
                assert!(
                    (metrics.correlation_insights.linear_correlations[[i, j]]
                        - metrics.correlation_insights.linear_correlations[[j, i]])
                    .abs()
                        < 1e-9
                );
            }
        }
    }

    #[test]
    fn test_shapiro_wilk_w_matches_scipy_reference() {
        // Regression test: `simplified_normality_test` used to compute a
        // skewness/kurtosis heuristic mislabeled as "Shapiro-Wilk"
        // (`shapiro_wilk_scores`); `shapiro_wilk_w` now computes the real W
        // statistic (Royston 1995 / AS R94), verified during development
        // against `scipy.stats.shapiro`, which reports W =
        // 0.9539016409629167 for this (non-constant, randomly generated)
        // sample.
        let data = Array1::from_vec(vec![
            6.232359134657199,
            6.534294537549542,
            4.321711505650686,
            4.054794733006444,
            2.2099454290316976,
            4.360094157736389,
            8.335752142958963,
            6.272440052377808,
            8.110637236669,
            5.746708182995274,
            6.1843089038412735,
            5.555979981285196,
        ]);
        let analyzer = AdvancedDatasetAnalyzer::new();
        let w = analyzer
            .shapiro_wilk_w(&data.view())
            .expect("Operation failed");
        assert!(
            (w - 0.9539016409629167).abs() < 1e-6,
            "expected W close to scipy's reference 0.9539016409629167, got {w}"
        );
    }

    #[test]
    fn test_shapiro_wilk_w_edge_cases() {
        let analyzer = AdvancedDatasetAnalyzer::new();

        // Constant data: not meaningfully non-normal (previously the old
        // heuristic returned 0.0 for this case, the opposite convention).
        let constant = Array1::from_vec(vec![5.0; 10]);
        let w_constant = analyzer
            .shapiro_wilk_w(&constant.view())
            .expect("Operation failed");
        assert!((w_constant - 1.0).abs() < 1e-12);

        // n < 3: undefined, neutral sentinel.
        let tiny = Array1::from_vec(vec![1.0, 2.0]);
        let w_tiny = analyzer
            .shapiro_wilk_w(&tiny.view())
            .expect("Operation failed");
        assert!((w_tiny - 0.5).abs() < 1e-12);
    }

    /// Regression test for the merged-in Jarque-Bera implementation: its
    /// final p-value-like transform used to be an arbitrary
    /// `(-jb_stat / 10.0).exp()` rescaling (no statistical meaning); it is
    /// now the exact chi-squared(df=2) survival function `exp(-JB/2)`.
    /// Verified against a hand-computed JB statistic on non-constant data.
    #[test]
    fn test_jarque_bera_matches_exact_chi_squared_transform() {
        let analyzer = AdvancedDatasetAnalyzer::new();
        let data = Array1::from_vec(vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 20.0, // one big outlier => skewed
        ]);
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let skew = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n;
        let excess_kurt = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n
            - 3.0;
        let expected_jb = (n / 6.0) * (skew.powi(2) + excess_kurt.powi(2) / 4.0);
        let expected_score = (-expected_jb / 2.0).exp();

        let score = analyzer
            .jarque_bera_test(data.view())
            .expect("Operation failed");
        assert!(
            (score - expected_score).abs() < 1e-9,
            "expected {expected_score}, got {score}"
        );
        // Skewed data with an outlier must not score as perfectly normal.
        assert!(score < 0.9);
    }

    /// Regression test for the merged-in Anderson-Darling implementation:
    /// clearly non-normal data (a strong outlier) must score meaningfully
    /// lower than data that is exactly normal-shaped-by-construction
    /// (evenly spaced quantiles of a standard normal).
    #[test]
    fn test_anderson_darling_distinguishes_normal_from_outlier_data() {
        let analyzer = AdvancedDatasetAnalyzer::new();

        // Quantiles of a standard normal at evenly spaced probabilities:
        // by construction, an almost perfect fit to normality.
        let normal = Normal::new(0.0, 1.0).expect("Operation failed");
        let n = 99;
        let near_normal: Vec<f64> = (1..=n)
            .map(|i| normal.inverse_cdf(i as f64 / (n as f64 + 1.0)))
            .collect();
        let near_normal_score = analyzer
            .anderson_darling_score(Array1::from_vec(near_normal).view())
            .expect("Operation failed");

        // Same size sample but with one huge outlier injected.
        let mut with_outlier: Vec<f64> = (1..=n)
            .map(|i| normal.inverse_cdf(i as f64 / (n as f64 + 1.0)))
            .collect();
        with_outlier[0] = 1000.0;
        let outlier_score = analyzer
            .anderson_darling_score(Array1::from_vec(with_outlier).view())
            .expect("Operation failed");

        assert!(
            near_normal_score > outlier_score,
            "near_normal_score={near_normal_score} should exceed outlier_score={outlier_score}"
        );
        assert!(near_normal_score > 0.5);
        assert!((0.0..=1.0).contains(&outlier_score));
    }

    /// Regression test for the merged-in complexity/entropy scores: they
    /// must depend on the actual data VALUES, not merely the sample count
    /// (the previous implementation of both was a pure function of
    /// `n_samples`, so two datasets of the same size but wildly different
    /// distributions -- e.g. constant vs. widely spread -- would have
    /// scored identically).
    #[test]
    fn test_complexity_and_entropy_depend_on_data_values() {
        let analyzer = AdvancedDatasetAnalyzer::new();

        let n = 200;
        // Same sample count, very different distributions.
        let constant = Array2::from_shape_vec((n, 1), vec![5.0; n]).expect("Operation failed");
        let spread =
            Array2::from_shape_vec((n, 1), (0..n).map(|i| (i as f64) * 0.1).collect::<Vec<_>>())
                .expect("Operation failed");

        let complexity_constant = analyzer
            .calculate_complexity_score(&constant)
            .expect("Operation failed");
        let complexity_spread = analyzer
            .calculate_complexity_score(&spread)
            .expect("Operation failed");
        assert!(
            complexity_spread > complexity_constant,
            "spread data ({complexity_spread}) must be more complex than constant data ({complexity_constant})"
        );
        assert_eq!(complexity_constant, 0.0);

        let entropy_constant = analyzer
            .calculate_entropy(&constant)
            .expect("Operation failed");
        let entropy_spread = analyzer
            .calculate_entropy(&spread)
            .expect("Operation failed");
        assert!(
            entropy_spread > entropy_constant,
            "spread data ({entropy_spread}) must have higher entropy than constant data ({entropy_constant})"
        );
    }

    /// Regression test for the merged-in Mahalanobis-based (diagonal
    /// covariance approximation) multivariate outlier score, which combines
    /// *all* features' squared z-scores into a single per-sample distance.
    ///
    /// # History
    ///
    /// The previous implementation counted outliers per *cell*: it checked
    /// each individual `(sample, feature)` value's own z-score against a
    /// fixed 3-sigma threshold independently, and reported the fraction of
    /// *cells* (not samples) exceeding it. Constructed here: a tight cluster
    /// plus one extra sample offset by 2.7 standard deviations on every one
    /// of 4 features -- comfortably under the old per-cell 3-sigma trigger
    /// on any single feature (verified numerically: per-feature z ~= 2.59),
    /// but its combined (summed-across-features) distance does exceed the
    /// new per-sample threshold, which is exactly the added detection power
    /// of aggregating across features instead of checking each in
    /// isolation.
    #[test]
    fn test_outlier_score_detects_combined_multi_feature_outlier() {
        let analyzer = AdvancedDatasetAnalyzer::new();

        let n = 100;
        let n_features = 4;
        // Small, non-constant baseline variation (period-7 sawtooth).
        let base_values: Vec<f64> = (0..n).map(|i| (((i % 7) as f64) - 3.0) * 0.1).collect();

        let base_mean = base_values.iter().sum::<f64>() / n as f64;
        let base_var = base_values
            .iter()
            .map(|&v| (v - base_mean).powi(2))
            .sum::<f64>()
            / (n as f64 - 1.0);
        let base_std = base_var.sqrt();
        // 2.7 sigma on every feature: well under the old per-cell z > 3.0
        // trigger, but jointly extreme once combined across 4 features.
        let outlier_value = base_mean + 2.7 * base_std;

        let mut rows: Vec<f64> = Vec::with_capacity((n + 1) * n_features);
        for &v in &base_values {
            for _ in 0..n_features {
                rows.push(v);
            }
        }
        for _ in 0..n_features {
            rows.push(outlier_value);
        }

        let data = Array2::from_shape_vec((n + 1, n_features), rows).expect("Operation failed");
        let dataset = Dataset::new(data, None);
        let metrics = analyzer
            .analyze_dataset_quality(&dataset)
            .expect("Operation failed");
        assert!(
            metrics.outlier_score > 0.0,
            "expected a nonzero outlier score, got {}",
            metrics.outlier_score
        );
    }
}
