//! API improvements for v1.0.0 release
//!
//! This module defines improved API patterns for better consistency and usability.

#![allow(dead_code)]

use crate::error::StatsResult;
use crate::tests::ttest::Alternative;
use scirs2_core::ndarray::{ArrayBase, Data, Ix1};
use scirs2_core::numeric::Float;

/// Standard correlation result that includes both coefficient and p-value
#[derive(Debug, Clone, Copy)]
pub struct CorrelationResult<F> {
    /// The correlation coefficient
    pub coefficient: F,
    /// The p-value (if computed)
    pub p_value: Option<F>,
}

impl<F: Float + std::fmt::Display> CorrelationResult<F> {
    /// Create a new correlation result with just the coefficient
    pub fn new(coefficient: F) -> Self {
        Self {
            coefficient,
            p_value: None,
        }
    }

    /// Create a new correlation result with coefficient and p-value
    pub fn with_p_value(_coefficient: F, pvalue: F) -> Self {
        Self {
            coefficient: _coefficient,
            p_value: Some(pvalue),
        }
    }
}

/// Correlation method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMethod {
    /// Pearson correlation coefficient
    Pearson,
    /// Spearman rank correlation
    Spearman,
    /// Kendall tau correlation
    KendallTau,
}

impl CorrelationMethod {
    /// Convert from string representation
    pub fn from_str(s: &str) -> StatsResult<Self> {
        match s.to_lowercase().as_str() {
            "pearson" => Ok(CorrelationMethod::Pearson),
            "spearman" => Ok(CorrelationMethod::Spearman),
            "kendall" | "kendall_tau" | "kendalltau" => Ok(CorrelationMethod::KendallTau),
            _ => Err(crate::error::StatsError::InvalidArgument(format!(
                "Invalid correlation method: '{}'",
                s
            ))),
        }
    }
}

/// Optimization hints for performance-critical operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationHint {
    /// Use the best available implementation (default)
    Auto,
    /// Force scalar implementation
    Scalar,
    /// Force SIMD implementation (if available)
    Simd,
    /// Force parallel implementation (if available)
    Parallel,
}

/// Configuration for statistical operations
#[derive(Debug, Clone)]
pub struct StatsConfig {
    /// Optimization hint for performance
    pub optimization: OptimizationHint,
    /// Whether to compute p-values
    pub compute_p_value: bool,
    /// Alternative hypothesis for tests
    pub alternative: Alternative,
}

impl Default for StatsConfig {
    fn default() -> Self {
        Self {
            optimization: OptimizationHint::Auto,
            compute_p_value: false,
            alternative: Alternative::TwoSided,
        }
    }
}

impl StatsConfig {
    /// Create a new configuration with p-value computation enabled
    pub fn with_p_value(mut self) -> Self {
        self.compute_p_value = true;
        self
    }

    /// Set the alternative hypothesis
    pub fn with_alternative(mut self, alternative: Alternative) -> Self {
        self.alternative = alternative;
        self
    }

    /// Set the optimization hint
    pub fn with_optimization(mut self, optimization: OptimizationHint) -> Self {
        self.optimization = optimization;
        self
    }
}

/// Improved correlation API that unifies pearson_r and pearsonr
pub trait CorrelationExt<F, D>
where
    F: Float + std::fmt::Display + std::iter::Sum + Send + Sync,
    D: Data<Elem = F>,
{
    /// Compute correlation with optional p-value based on configuration
    fn correlation(
        &self,
        other: &ArrayBase<D, Ix1>,
        method: CorrelationMethod,
        config: Option<StatsConfig>,
    ) -> StatsResult<CorrelationResult<F>>;

    /// Compute Pearson correlation (convenience method)
    fn pearson(&self, other: &ArrayBase<D, Ix1>) -> StatsResult<F> {
        self.correlation(other, CorrelationMethod::Pearson, None)
            .map(|r| r.coefficient)
    }

    /// Compute Spearman correlation (convenience method)
    fn spearman(&self, other: &ArrayBase<D, Ix1>) -> StatsResult<F> {
        self.correlation(other, CorrelationMethod::Spearman, None)
            .map(|r| r.coefficient)
    }

    /// Compute Kendall tau correlation (convenience method)
    fn kendall(&self, other: &ArrayBase<D, Ix1>) -> StatsResult<F> {
        self.correlation(other, CorrelationMethod::KendallTau, None)
            .map(|r| r.coefficient)
    }
}

/// Builder pattern for complex statistical operations
pub struct StatsBuilder<F> {
    data: Option<Vec<F>>,
    config: StatsConfig,
}

impl<F: Float + std::fmt::Display + std::iter::Sum + Send + Sync> Default for StatsBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float + std::fmt::Display + std::iter::Sum + Send + Sync> StatsBuilder<F> {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            data: None,
            config: StatsConfig::default(),
        }
    }

    /// Set the data with validation
    pub fn data(mut self, data: Vec<F>) -> StatsResult<Self> {
        // Check if data is empty
        if data.is_empty() {
            return Err(crate::error::StatsError::invalid_argument(
                "Data cannot be empty",
            ));
        }

        self.data = Some(data);
        Ok(self)
    }

    /// Set data without validation (for performance-critical paths)
    pub fn data_unchecked(mut self, data: Vec<F>) -> Self {
        self.data = Some(data);
        self
    }

    /// Enable p-value computation
    pub fn with_p_value(mut self) -> Self {
        self.config.compute_p_value = true;
        self
    }

    /// Set the alternative hypothesis
    pub fn alternative(mut self, alt: Alternative) -> Self {
        self.config.alternative = alt;
        self
    }

    /// Set optimization hint
    pub fn optimization(mut self, opt: OptimizationHint) -> Self {
        self.config.optimization = opt;
        self
    }

    /// Validate the current configuration
    pub fn validate(&self) -> StatsResult<()> {
        if self.data.is_none() {
            return Err(crate::error::StatsError::invalid_argument(
                "No data provided to builder",
            ));
        }

        if let Some(ref data) = self.data {
            if data.is_empty() {
                return Err(crate::error::StatsError::invalid_argument(
                    "Data cannot be empty",
                ));
            }
        }

        Ok(())
    }

    /// Get a reference to the data
    pub fn getdata(&self) -> Option<&Vec<F>> {
        self.data.as_ref()
    }

    /// Get the configuration
    pub fn get_config(&self) -> &StatsConfig {
        &self.config
    }
}

/// Improved test result that standardizes output across all tests
#[derive(Debug, Clone)]
pub struct TestResult<F> {
    /// The test statistic
    pub statistic: F,
    /// The p-value
    pub p_value: F,
    /// Degrees of freedom (if applicable)
    pub df: Option<F>,
    /// Effect size (if applicable)
    pub effectsize: Option<F>,
    /// Confidence interval (if applicable)
    pub confidence_interval: Option<(F, F)>,
}

impl<F: Float + std::fmt::Display> TestResult<F> {
    /// Create a basic test result
    pub fn new(_statistic: F, pvalue: F) -> Self {
        Self {
            statistic: _statistic,
            p_value: pvalue,
            df: None,
            effectsize: None,
            confidence_interval: None,
        }
    }

    /// Add degrees of freedom
    pub fn with_df(mut self, df: F) -> Self {
        self.df = Some(df);
        self
    }

    /// Add effect size
    pub fn with_effectsize(mut self, effectsize: F) -> Self {
        self.effectsize = Some(effectsize);
        self
    }

    /// Add confidence interval
    pub fn with_confidence_interval(mut self, lower: F, upper: F) -> Self {
        self.confidence_interval = Some((lower, upper));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_method_from_str() {
        assert_eq!(
            CorrelationMethod::from_str("pearson").unwrap(),
            CorrelationMethod::Pearson
        );
        assert_eq!(
            CorrelationMethod::from_str("spearman").unwrap(),
            CorrelationMethod::Spearman
        );
        assert_eq!(
            CorrelationMethod::from_str("kendall").unwrap(),
            CorrelationMethod::KendallTau
        );
        assert!(CorrelationMethod::from_str("invalid").is_err());
    }

    #[test]
    fn test_stats_config_builder() {
        let config = StatsConfig::default()
            .with_p_value()
            .with_alternative(Alternative::Greater);

        assert!(config.compute_p_value);
        assert_eq!(config.alternative, Alternative::Greater);
        assert_eq!(config.optimization, OptimizationHint::Auto);
    }
}
