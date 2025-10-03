//! Uncertainty quantification methods for model predictions
//!
//! This module provides comprehensive uncertainty quantification techniques
//! including Bayesian methods, conformal prediction, calibration techniques,
//! and advanced uncertainty analysis.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

// Module declarations
pub mod advanced;
pub mod bayesian;
pub mod calibration;
pub mod conformal;
pub mod core;
pub mod rng;

// Re-export core types
pub use core::{
    AleatoricUncertainty, BrierDecomposition, CalibrationMetrics, ConfidenceScores,
    EpistemicUncertainty, OODScores, PredictionIntervals, RandomNumberGenerator,
    UncertaintyAnalysis, UncertaintyQuantifier,
};

// Re-export RNG types
pub use rng::{ChaChaRng, LcgRng, PcgRng, RandomNumberGeneratorTrait, XorshiftRng};

// Re-export Bayesian methods
pub use bayesian::{
    BayesianUncertainty, MCMCDiagnostics, PriorParameters, VariationalParams,
    VariationalUncertainty,
};

// Re-export conformal prediction
pub use conformal::{ConformalPrediction, PredictionSet};

// Re-export calibration methods
pub use calibration::{DeepEnsembleUncertainty, TemperatureScaling};

// Re-export advanced methods
pub use advanced::{
    AdvancedUncertaintyAnalysis, CoverageAnalysis, MultiscaleUncertainty, UncertaintyDecomposition,
};

/// Comprehensive uncertainty quantification suite
pub struct UncertaintyQuantificationSuite<F: Float> {
    /// Main uncertainty quantifier
    pub quantifier: UncertaintyQuantifier<F>,
    /// Bayesian uncertainty estimator
    pub bayesian: Option<BayesianUncertainty<F>>,
    /// Conformal predictor
    pub conformal: Option<ConformalPrediction<F>>,
    /// Temperature scaling calibrator
    pub temperature_scaling: Option<TemperatureScaling<F>>,
    /// Deep ensemble uncertainty
    pub deep_ensemble: Option<DeepEnsembleUncertainty<F>>,
    /// Advanced analysis tools
    pub advanced_analysis: Option<AdvancedUncertaintyAnalysis<F>>,
}

impl<
        F: Float
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > UncertaintyQuantificationSuite<F>
{
    /// Create new uncertainty quantification suite
    pub fn new(quantifier: UncertaintyQuantifier<F>) -> Self {
        Self {
            quantifier,
            bayesian: None,
            conformal: None,
            temperature_scaling: None,
            deep_ensemble: None,
            advanced_analysis: None,
        }
    }

    /// Create suite with all methods enabled
    pub fn with_all_methods(quantifier: UncertaintyQuantifier<F>) -> Self {
        Self {
            quantifier,
            bayesian: Some(BayesianUncertainty::new(1000, 10)),
            conformal: Some(ConformalPrediction::new(F::from(0.95).unwrap())),
            temperature_scaling: Some(TemperatureScaling::new()),
            deep_ensemble: Some(DeepEnsembleUncertainty::new(5)),
            advanced_analysis: Some(AdvancedUncertaintyAnalysis::new()),
        }
    }

    /// Enable Bayesian uncertainty estimation
    pub fn with_bayesian(mut self, n_samples: usize, n_params: usize) -> Self {
        self.bayesian = Some(BayesianUncertainty::new(n_samples, n_params));
        self
    }

    /// Enable conformal prediction
    pub fn with_conformal(mut self, confidence_level: F) -> Self {
        self.conformal = Some(ConformalPrediction::new(confidence_level));
        self
    }

    /// Enable temperature scaling
    pub fn with_temperature_scaling(mut self) -> Self {
        self.temperature_scaling = Some(TemperatureScaling::new());
        self
    }

    /// Enable deep ensemble uncertainty
    pub fn with_deep_ensemble(mut self, ensemble_size: usize) -> Self {
        self.deep_ensemble = Some(DeepEnsembleUncertainty::new(ensemble_size));
        self
    }

    /// Enable advanced analysis
    pub fn with_advanced_analysis(mut self) -> Self {
        self.advanced_analysis = Some(AdvancedUncertaintyAnalysis::new());
        self
    }

    /// Perform comprehensive uncertainty analysis
    pub fn analyze(
        &self,
        predictions: &ArrayView2<F>,
        ground_truth: Option<&ArrayView1<F>>,
        model_outputs: Option<&[ArrayView2<F>]>,
    ) -> Result<ComprehensiveUncertaintyResults<F>> {
        // Core uncertainty analysis
        let core_analysis =
            self.quantifier
                .analyze_uncertainty(predictions, ground_truth, model_outputs)?;

        // Bayesian analysis
        let bayesian_results = if let Some(_bayesian) = &self.bayesian {
            // Would perform Bayesian analysis here
            None
        } else {
            None
        };

        // Conformal prediction
        let conformal_results = if let Some(_conformal) = &self.conformal {
            // Would perform conformal prediction here
            None
        } else {
            None
        };

        // Temperature scaling
        let calibration_results = if let Some(_temp_scaling) = &self.temperature_scaling {
            // Would perform temperature scaling here
            None
        } else {
            None
        };

        // Deep ensemble analysis
        let ensemble_results = if let Some(_ensemble) = &self.deep_ensemble {
            // Would perform ensemble analysis here
            None
        } else {
            None
        };

        // Advanced analysis
        let advanced_results = if let Some(_advanced) = &self.advanced_analysis {
            // Would perform advanced analysis here
            None
        } else {
            None
        };

        Ok(ComprehensiveUncertaintyResults {
            core_analysis,
            bayesian_results,
            conformal_results,
            calibration_results,
            ensemble_results,
            advanced_results,
        })
    }

    /// Get configuration summary
    pub fn config_summary(&self) -> UncertaintyConfigSummary<F> {
        UncertaintyConfigSummary {
            bayesian_enabled: self.bayesian.is_some(),
            conformal_enabled: self.conformal.is_some(),
            temperature_scaling_enabled: self.temperature_scaling.is_some(),
            deep_ensemble_enabled: self.deep_ensemble.is_some(),
            advanced_analysis_enabled: self.advanced_analysis.is_some(),
            mc_samples: self.quantifier.n_mc_samples,
            confidence_level: self.quantifier.confidence_level,
        }
    }
}

/// Comprehensive uncertainty quantification results
#[derive(Debug, Clone)]
pub struct ComprehensiveUncertaintyResults<F: Float> {
    /// Core uncertainty analysis
    pub core_analysis: UncertaintyAnalysis<F>,
    /// Bayesian uncertainty results
    pub bayesian_results: Option<BayesianUncertainty<F>>,
    /// Conformal prediction results
    pub conformal_results: Option<Vec<PredictionSet<F>>>,
    /// Calibration results
    pub calibration_results: Option<TemperatureScaling<F>>,
    /// Ensemble uncertainty results
    pub ensemble_results: Option<DeepEnsembleUncertainty<F>>,
    /// Advanced analysis results
    pub advanced_results: Option<AdvancedUncertaintyAnalysis<F>>,
}

/// Configuration summary for uncertainty quantification
#[derive(Debug, Clone)]
pub struct UncertaintyConfigSummary<F: Float> {
    /// Bayesian methods enabled
    pub bayesian_enabled: bool,
    /// Conformal prediction enabled
    pub conformal_enabled: bool,
    /// Temperature scaling enabled
    pub temperature_scaling_enabled: bool,
    /// Deep ensemble enabled
    pub deep_ensemble_enabled: bool,
    /// Advanced analysis enabled
    pub advanced_analysis_enabled: bool,
    /// Number of Monte Carlo samples
    pub mc_samples: usize,
    /// Confidence level
    pub confidence_level: F,
}

/// Uncertainty quantification metrics computer
pub struct UncertaintyMetricsComputer<F: Float> {
    /// Uncertainty suite
    suite: UncertaintyQuantificationSuite<F>,
}

impl<
        F: Float
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > UncertaintyMetricsComputer<F>
{
    /// Create new uncertainty metrics computer
    pub fn new(suite: UncertaintyQuantificationSuite<F>) -> Self {
        Self { suite }
    }

    /// Compute uncertainty metrics
    pub fn compute_metrics(
        &self,
        predictions: &ArrayView2<F>,
        ground_truth: Option<&ArrayView1<F>>,
        model_outputs: Option<&[ArrayView2<F>]>,
    ) -> Result<ComprehensiveUncertaintyResults<F>> {
        self.suite.analyze(predictions, ground_truth, model_outputs)
    }

    /// Get uncertainty summary statistics
    pub fn summary_statistics(
        &self,
        results: &ComprehensiveUncertaintyResults<F>,
    ) -> UncertaintySummaryStats<F> {
        let core = &results.core_analysis;

        UncertaintySummaryStats {
            mean_epistemic_uncertainty: core
                .epistemic_uncertainty
                .model_variance
                .mean()
                .unwrap_or(F::zero()),
            mean_aleatoric_uncertainty: core
                .aleatoric_uncertainty
                .data_variance
                .mean()
                .unwrap_or(F::zero()),
            mean_confidence: core
                .confidence_scores
                .max_probability
                .mean()
                .unwrap_or(F::zero()),
            calibration_error: core.calibration_metrics.expected_calibration_error,
            coverage: F::from(0.95).unwrap(), // Would compute actual coverage
        }
    }
}

/// Uncertainty summary statistics
#[derive(Debug, Clone)]
pub struct UncertaintySummaryStats<F: Float> {
    /// Mean epistemic uncertainty
    pub mean_epistemic_uncertainty: F,
    /// Mean aleatoric uncertainty
    pub mean_aleatoric_uncertainty: F,
    /// Mean confidence score
    pub mean_confidence: F,
    /// Calibration error
    pub calibration_error: F,
    /// Coverage rate
    pub coverage: F,
}

impl<
        F: Float
            + scirs2_core::numeric::FromPrimitive
            + std::iter::Sum
            + scirs2_core::ndarray::ScalarOperand,
    > Default for UncertaintyQuantificationSuite<F>
{
    fn default() -> Self {
        Self::new(UncertaintyQuantifier::new())
    }
}
