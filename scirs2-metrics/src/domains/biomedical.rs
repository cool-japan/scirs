//! Biomedical and healthcare domain metrics
//!
//! This module provides specialized metrics for biomedical applications including:
//! - Clinical trial analysis and endpoint evaluation
//! - Drug discovery and development metrics
//! - Medical imaging and diagnostic accuracy
//! - Genomics and bioinformatics analysis
//! - Epidemiological and population health metrics
//! - Medical device performance evaluation
//! - Patient outcome and survival analysis

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Comprehensive biomedical metrics suite
#[derive(Debug)]
pub struct BiomedicalSuite {
    /// Clinical trial metrics
    pub clinical_trial_metrics: ClinicalTrialMetrics,
    /// Drug discovery metrics
    pub drug_discovery_metrics: DrugDiscoveryMetrics,
    /// Medical imaging metrics
    pub medical_imaging_metrics: MedicalImagingMetrics,
    /// Genomics analysis metrics
    pub genomics_metrics: GenomicsMetrics,
    /// Epidemiological metrics
    pub epidemiology_metrics: EpidemiologyMetrics,
    /// Survival analysis metrics
    pub survival_metrics: SurvivalAnalysisMetrics,
    /// Biomarker evaluation metrics
    pub biomarker_metrics: BiomarkerMetrics,
}

impl Default for BiomedicalSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl BiomedicalSuite {
    /// Create new biomedical metrics suite
    pub fn new() -> Self {
        Self {
            clinical_trial_metrics: ClinicalTrialMetrics::new(),
            drug_discovery_metrics: DrugDiscoveryMetrics::new(),
            medical_imaging_metrics: MedicalImagingMetrics::new(),
            genomics_metrics: GenomicsMetrics::new(),
            epidemiology_metrics: EpidemiologyMetrics::new(),
            survival_metrics: SurvivalAnalysisMetrics::new(),
            biomarker_metrics: BiomarkerMetrics::new(),
        }
    }

    /// Get clinical trial metrics
    pub fn clinical_trial(&self) -> &ClinicalTrialMetrics {
        &self.clinical_trial_metrics
    }

    /// Get drug discovery metrics
    pub fn drug_discovery(&self) -> &DrugDiscoveryMetrics {
        &self.drug_discovery_metrics
    }

    /// Get medical imaging metrics
    pub fn medical_imaging(&self) -> &MedicalImagingMetrics {
        &self.medical_imaging_metrics
    }

    /// Get genomics metrics
    pub fn genomics(&self) -> &GenomicsMetrics {
        &self.genomics_metrics
    }

    /// Get epidemiological metrics
    pub fn epidemiology(&self) -> &EpidemiologyMetrics {
        &self.epidemiology_metrics
    }

    /// Get survival analysis metrics
    pub fn survival_analysis(&self) -> &SurvivalAnalysisMetrics {
        &self.survival_metrics
    }

    /// Get biomarker evaluation metrics
    pub fn biomarker(&self) -> &BiomarkerMetrics {
        &self.biomarker_metrics
    }
}

impl DomainMetrics for BiomedicalSuite {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Biomedical & Healthcare"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "sensitivity",
            "specificity",
            "positive_predictive_value",
            "negative_predictive_value",
            "likelihood_ratio_positive",
            "likelihood_ratio_negative",
            "diagnostic_odds_ratio",
            "number_needed_to_treat",
            "number_needed_to_harm",
            "area_under_roc",
            "concordance_index",
            "hazard_ratio",
            "survival_rate",
            "time_to_event",
            "progression_free_survival",
            "overall_survival",
            "quality_adjusted_life_years",
            "disability_adjusted_life_years",
            "dice_coefficient",
            "jaccard_index",
            "hausdorff_distance",
            "volumetric_overlap",
            "genomic_concordance",
            "variant_calling_accuracy",
            "population_attributable_risk",
            "incidence_rate_ratio",
            "odds_ratio",
            "relative_risk",
            "attributable_risk",
            "biomarker_auc",
            "biomarker_sensitivity",
            "biomarker_specificity",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "sensitivity",
            "True positive rate (recall) - ability to correctly identify positive cases",
        );
        descriptions.insert(
            "specificity",
            "True negative rate - ability to correctly identify negative cases",
        );
        descriptions.insert(
            "positive_predictive_value",
            "Precision - proportion of positive predictions that are correct",
        );
        descriptions.insert(
            "negative_predictive_value",
            "Proportion of negative predictions that are correct",
        );
        descriptions.insert(
            "likelihood_ratio_positive",
            "Ratio of true positive rate to false positive rate",
        );
        descriptions.insert(
            "likelihood_ratio_negative",
            "Ratio of false negative rate to true negative rate",
        );
        descriptions.insert(
            "diagnostic_odds_ratio",
            "Effectiveness of a diagnostic test",
        );
        descriptions.insert(
            "number_needed_to_treat",
            "Number of patients to treat for one to benefit",
        );
        descriptions.insert(
            "number_needed_to_harm",
            "Number of patients to treat for one to be harmed",
        );
        descriptions.insert(
            "concordance_index",
            "Probability that predictions are correctly ordered",
        );
        descriptions.insert(
            "hazard_ratio",
            "Relative risk of an event occurring over time",
        );
        descriptions.insert(
            "quality_adjusted_life_years",
            "Measure combining length and quality of life",
        );
        descriptions.insert(
            "dice_coefficient",
            "Overlap measure for segmentation accuracy",
        );
        descriptions.insert(
            "hausdorff_distance",
            "Maximum distance between two sets of points",
        );
        descriptions.insert(
            "genomic_concordance",
            "Agreement between genomic variant calls",
        );
        descriptions.insert(
            "population_attributable_risk",
            "Proportion of disease incidence attributable to exposure",
        );
        descriptions.insert(
            "biomarker_auc",
            "Area under ROC curve for biomarker performance",
        );
        descriptions
    }
}

/// Clinical trial analysis metrics
#[derive(Debug, Clone)]
pub struct ClinicalTrialMetrics {
    /// Primary endpoint results
    pub primary_endpoints: HashMap<String, f64>,
    /// Secondary endpoint results  
    pub secondary_endpoints: HashMap<String, f64>,
    /// Safety endpoint results
    pub safety_endpoints: HashMap<String, f64>,
}

impl ClinicalTrialMetrics {
    pub fn new() -> Self {
        Self {
            primary_endpoints: HashMap::new(),
            secondary_endpoints: HashMap::new(),
            safety_endpoints: HashMap::new(),
        }
    }

    /// Compute treatment efficacy from clinical trial data
    pub fn compute_efficacy<F>(
        &self,
        treatment_outcomes: &Array1<F>,
        control_outcomes: &Array1<F>,
        outcome_type: EfficacyOutcome,
    ) -> Result<ClinicalEfficacyResult<F>>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let treatment_mean = treatment_outcomes.iter().cloned().sum::<F>()
            / F::from(treatment_outcomes.len()).unwrap();
        let control_mean =
            control_outcomes.iter().cloned().sum::<F>() / F::from(control_outcomes.len()).unwrap();

        let effect_size = match outcome_type {
            EfficacyOutcome::Continuous => {
                // Cohen's d for continuous _outcomes
                let pooled_std = self.compute_pooled_std(treatment_outcomes, control_outcomes)?;
                (treatment_mean - control_mean) / pooled_std
            }
            EfficacyOutcome::Binary => {
                // Risk difference for binary _outcomes
                treatment_mean - control_mean
            }
            EfficacyOutcome::TimeToEvent => {
                // Hazard ratio (simplified calculation)
                if control_mean > F::zero() {
                    treatment_mean / control_mean
                } else {
                    F::one()
                }
            }
        };

        // Calculate confidence interval (simplified)
        let standard_error = self.compute_standard_error(treatment_outcomes, control_outcomes)?;
        let margin_of_error = F::from(1.96).unwrap() * standard_error;

        Ok(ClinicalEfficacyResult {
            effect_size,
            confidence_interval_lower: effect_size - margin_of_error,
            confidence_interval_upper: effect_size + margin_of_error,
            p_value: self.compute_p_value(treatment_outcomes, control_outcomes)?,
            treatment_mean,
            control_mean,
            outcome_type,
        })
    }

    /// Compute Number Needed to Treat (NNT)
    pub fn number_needed_to_treat<F>(
        &self,
        treatment_success_rate: F,
        control_success_rate: F,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let absolute_risk_reduction = treatment_success_rate - control_success_rate;
        if absolute_risk_reduction > F::zero() {
            Ok(F::one() / absolute_risk_reduction)
        } else {
            Err(MetricsError::InvalidInput(
                "Treatment success _rate must be higher than control".to_string(),
            ))
        }
    }

    /// Compute Number Needed to Harm (NNH)
    pub fn number_needed_to_harm<F>(
        &self,
        treatment_adverse_rate: F,
        control_adverse_rate: F,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let absolute_risk_increase = treatment_adverse_rate - control_adverse_rate;
        if absolute_risk_increase > F::zero() {
            Ok(F::one() / absolute_risk_increase)
        } else {
            Err(MetricsError::InvalidInput(
                "Treatment adverse _rate must be higher than control".to_string(),
            ))
        }
    }

    /// Helper method to compute pooled standard deviation
    fn compute_pooled_std<F>(&self, group1: &Array1<F>, group2: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let n1 = F::from(group1.len()).unwrap();
        let n2 = F::from(group2.len()).unwrap();

        let mean1 = group1.iter().cloned().sum::<F>() / n1;
        let mean2 = group2.iter().cloned().sum::<F>() / n2;

        let var1 = group1.iter().map(|&x| (x - mean1) * (x - mean1)).sum::<F>() / (n1 - F::one());
        let var2 = group2.iter().map(|&x| (x - mean2) * (x - mean2)).sum::<F>() / (n2 - F::one());

        let pooled_var =
            ((n1 - F::one()) * var1 + (n2 - F::one()) * var2) / (n1 + n2 - F::from(2).unwrap());
        Ok(pooled_var.sqrt())
    }

    /// Helper method to compute standard error
    fn compute_standard_error<F>(&self, group1: &Array1<F>, group2: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let n1 = F::from(group1.len()).unwrap();
        let n2 = F::from(group2.len()).unwrap();
        let pooled_std = self.compute_pooled_std(group1, group2)?;

        Ok(pooled_std * (F::one() / n1 + F::one() / n2).sqrt())
    }

    /// Helper method to compute p-value (simplified t-test)
    fn compute_p_value<F>(&self, group1: &Array1<F>, group2: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        // Simplified p-value calculation
        // In a real implementation, would use proper statistical test
        let effect_size = self.compute_pooled_std(group1, group2)?;
        if effect_size > F::from(2.0).unwrap() {
            Ok(F::from(0.05).unwrap())
        } else if effect_size > F::one() {
            Ok(F::from(0.1).unwrap())
        } else {
            Ok(F::from(0.5).unwrap())
        }
    }
}

impl Default for ClinicalTrialMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Drug discovery and development metrics
#[derive(Debug, Clone)]
pub struct DrugDiscoveryMetrics {
    /// Molecular property predictions
    pub molecular_properties: HashMap<String, f64>,
    /// ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) predictions
    pub admet_predictions: HashMap<String, f64>,
    /// Target engagement metrics
    pub target_engagement: HashMap<String, f64>,
}

impl DrugDiscoveryMetrics {
    pub fn new() -> Self {
        Self {
            molecular_properties: HashMap::new(),
            admet_predictions: HashMap::new(),
            target_engagement: HashMap::new(),
        }
    }

    /// Compute drug-target binding affinity metrics
    pub fn binding_affinity_metrics<F>(
        &self,
        predicted_affinities: &Array1<F>,
        experimental_affinities: &Array1<F>,
    ) -> Result<DrugTargetMetrics<F>>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if predicted_affinities.len() != experimental_affinities.len() {
            return Err(MetricsError::InvalidInput(
                "Predicted and experimental arrays must have same length".to_string(),
            ));
        }

        // Pearson correlation coefficient
        let correlation =
            self.compute_correlation(predicted_affinities, experimental_affinities)?;

        // Root Mean Square Error
        let rmse = {
            let mse = predicted_affinities
                .iter()
                .zip(experimental_affinities.iter())
                .map(|(&p, &e)| (p - e) * (p - e))
                .sum::<F>()
                / F::from(predicted_affinities.len()).unwrap();
            mse.sqrt()
        };

        // Mean Absolute Error
        let mae = predicted_affinities
            .iter()
            .zip(experimental_affinities.iter())
            .map(|(&p, &e)| (p - e).abs())
            .sum::<F>()
            / F::from(predicted_affinities.len()).unwrap();

        // R-squared
        let mean_exp = experimental_affinities.iter().cloned().sum::<F>()
            / F::from(experimental_affinities.len()).unwrap();
        let ss_tot = experimental_affinities
            .iter()
            .map(|&e| (e - mean_exp) * (e - mean_exp))
            .sum::<F>();
        let ss_res = predicted_affinities
            .iter()
            .zip(experimental_affinities.iter())
            .map(|(&p, &e)| (e - p) * (e - p))
            .sum::<F>();
        let r_squared = if ss_tot > F::zero() {
            F::one() - ss_res / ss_tot
        } else {
            F::zero()
        };

        Ok(DrugTargetMetrics {
            correlation,
            rmse,
            mae,
            r_squared,
            concordance_index: self
                .compute_concordance_index(predicted_affinities, experimental_affinities)?,
        })
    }

    /// Compute Tanimoto similarity for molecular fingerprints
    pub fn tanimoto_similarity(
        &self,
        fingerprint1: &Array1<u8>,
        fingerprint2: &Array1<u8>,
    ) -> Result<f64> {
        if fingerprint1.len() != fingerprint2.len() {
            return Err(MetricsError::InvalidInput(
                "Fingerprints must have same length".to_string(),
            ));
        }

        let intersection = fingerprint1
            .iter()
            .zip(fingerprint2.iter())
            .map(|(&a, &b)| (a & b) as u32)
            .sum::<u32>();

        let union = fingerprint1
            .iter()
            .zip(fingerprint2.iter())
            .map(|(&a, &b)| (a | b) as u32)
            .sum::<u32>();

        if union == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f64 / union as f64)
        }
    }

    /// Helper method to compute correlation
    fn compute_correlation<F>(&self, x: &Array1<F>, y: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let n = F::from(x.len()).unwrap();
        let mean_x = x.iter().cloned().sum::<F>() / n;
        let mean_y = y.iter().cloned().sum::<F>() / n;

        let numerator = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<F>();

        let sum_sq_x = x.iter().map(|&xi| (xi - mean_x) * (xi - mean_x)).sum::<F>();
        let sum_sq_y = y.iter().map(|&yi| (yi - mean_y) * (yi - mean_y)).sum::<F>();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > F::zero() {
            Ok(numerator / denominator)
        } else {
            Ok(F::zero())
        }
    }

    /// Helper method to compute concordance index
    fn compute_concordance_index<F>(&self, predicted: &Array1<F>, actual: &Array1<F>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let mut concordant = 0;
        let mut total = 0;

        for i in 0..predicted.len() {
            for j in (i + 1)..predicted.len() {
                let pred_diff = predicted[i] - predicted[j];
                let actual_diff = actual[i] - actual[j];

                if (pred_diff > F::zero() && actual_diff > F::zero())
                    || (pred_diff < F::zero() && actual_diff < F::zero())
                {
                    concordant += 1;
                }
                total += 1;
            }
        }

        if total > 0 {
            Ok(F::from(concordant).unwrap() / F::from(total).unwrap())
        } else {
            Ok(F::from(0.5).unwrap())
        }
    }
}

impl Default for DrugDiscoveryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Medical imaging analysis metrics
#[derive(Debug, Clone)]
pub struct MedicalImagingMetrics {
    /// Segmentation accuracy metrics
    pub segmentation_metrics: HashMap<String, f64>,
    /// Diagnostic accuracy metrics
    pub diagnostic_metrics: HashMap<String, f64>,
    /// Image quality metrics
    pub quality_metrics: HashMap<String, f64>,
}

impl MedicalImagingMetrics {
    pub fn new() -> Self {
        Self {
            segmentation_metrics: HashMap::new(),
            diagnostic_metrics: HashMap::new(),
            quality_metrics: HashMap::new(),
        }
    }

    /// Compute Dice coefficient for segmentation accuracy
    pub fn dice_coefficient<F>(
        &self,
        predicted_mask: &Array2<F>,
        ground_truth_mask: &Array2<F>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if predicted_mask.shape() != ground_truth_mask.shape() {
            return Err(MetricsError::InvalidInput(
                "Masks must have the same shape".to_string(),
            ));
        }

        let intersection = predicted_mask
            .iter()
            .zip(ground_truth_mask.iter())
            .map(|(&p, &g)| {
                if p > F::zero() && g > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .sum::<F>();

        let predicted_sum = predicted_mask
            .iter()
            .map(|&p| if p > F::zero() { F::one() } else { F::zero() })
            .sum::<F>();

        let ground_truth_sum = ground_truth_mask
            .iter()
            .map(|&g| if g > F::zero() { F::one() } else { F::zero() })
            .sum::<F>();

        let denominator = predicted_sum + ground_truth_sum;

        if denominator > F::zero() {
            Ok(F::from(2.0).unwrap() * intersection / denominator)
        } else {
            Ok(F::one()) // Perfect match when both masks are empty
        }
    }

    /// Compute Jaccard index (Intersection over Union)
    pub fn jaccard_index<F>(
        &self,
        predicted_mask: &Array2<F>,
        ground_truth_mask: &Array2<F>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if predicted_mask.shape() != ground_truth_mask.shape() {
            return Err(MetricsError::InvalidInput(
                "Masks must have the same shape".to_string(),
            ));
        }

        let intersection = predicted_mask
            .iter()
            .zip(ground_truth_mask.iter())
            .map(|(&p, &g)| {
                if p > F::zero() && g > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .sum::<F>();

        let union = predicted_mask
            .iter()
            .zip(ground_truth_mask.iter())
            .map(|(&p, &g)| {
                if p > F::zero() || g > F::zero() {
                    F::one()
                } else {
                    F::zero()
                }
            })
            .sum::<F>();

        if union > F::zero() {
            Ok(intersection / union)
        } else {
            Ok(F::one()) // Perfect match when both masks are empty
        }
    }

    /// Compute Hausdorff distance between two segmentation masks
    pub fn hausdorff_distance<F>(
        &self,
        predicted_mask: &Array2<F>,
        ground_truth_mask: &Array2<F>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if predicted_mask.shape() != ground_truth_mask.shape() {
            return Err(MetricsError::InvalidInput(
                "Masks must have the same shape".to_string(),
            ));
        }

        // Extract boundary points (simplified implementation)
        let pred_points = self.extract_boundary_points(predicted_mask);
        let gt_points = self.extract_boundary_points(ground_truth_mask);

        if pred_points.is_empty() || gt_points.is_empty() {
            return Ok(F::zero());
        }

        // Compute directed Hausdorff distances
        let h1: F = self.directed_hausdorff_distance(&pred_points, &gt_points);
        let h2: F = self.directed_hausdorff_distance(&gt_points, &pred_points);

        Ok(h1.max(h2))
    }

    /// Helper method to extract boundary points from a mask
    fn extract_boundary_points<F>(&self, mask: &Array2<F>) -> Vec<(usize, usize)>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let mut boundary_points = Vec::new();
        let (rows, cols) = mask.dim();

        for i in 0..rows {
            for j in 0..cols {
                if mask[[i, j]] > F::zero() {
                    // Check if this is a boundary point
                    let mut is_boundary = false;
                    for di in -1i32..=1 {
                        for dj in -1i32..=1 {
                            let ni = i as i32 + di;
                            let nj = j as i32 + dj;
                            if ni >= 0 && (ni as usize) < rows && nj >= 0 && (nj as usize) < cols {
                                if mask[[ni as usize, nj as usize]] <= F::zero() {
                                    is_boundary = true;
                                    break;
                                }
                            } else {
                                is_boundary = true;
                                break;
                            }
                        }
                        if is_boundary {
                            break;
                        }
                    }
                    if is_boundary {
                        boundary_points.push((i, j));
                    }
                }
            }
        }

        boundary_points
    }

    /// Helper method to compute directed Hausdorff distance
    fn directed_hausdorff_distance<F>(
        &self,
        points1: &[(usize, usize)],
        points2: &[(usize, usize)],
    ) -> F
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        let mut max_min_dist = F::zero();

        for &(x1, y1) in points1 {
            let mut min_dist = F::infinity();
            for &(x2, y2) in points2 {
                let dx = F::from(x1 as i32 - x2 as i32).unwrap();
                let dy = F::from(y1 as i32 - y2 as i32).unwrap();
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            if min_dist > max_min_dist {
                max_min_dist = min_dist;
            }
        }

        max_min_dist
    }
}

impl Default for MedicalImagingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Genomics and bioinformatics metrics
#[derive(Debug, Clone)]
pub struct GenomicsMetrics;

impl GenomicsMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Compute genomic variant calling accuracy
    pub fn variant_calling_accuracy(
        &self,
        predicted_variants: &[(usize, char, char)], // (position, ref, alt)
        true_variants: &[(usize, char, char)],
    ) -> Result<GenomicAccuracyMetrics> {
        let mut true_positives = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        let pred_set: std::collections::HashSet<_> = predicted_variants.iter().collect();
        let true_set: std::collections::HashSet<_> = true_variants.iter().collect();

        // True positives: _variants in both sets
        for variant in &pred_set {
            if true_set.contains(variant) {
                true_positives += 1;
            } else {
                false_positives += 1;
            }
        }

        // False negatives: _variants in true set but not in predicted
        for variant in &true_set {
            if !pred_set.contains(variant) {
                false_negatives += 1;
            }
        }

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(GenomicAccuracyMetrics {
            precision,
            recall,
            f1_score,
            true_positives,
            false_positives,
            false_negatives,
        })
    }
}

impl Default for GenomicsMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Epidemiological and population health metrics
#[derive(Debug, Clone)]
pub struct EpidemiologyMetrics;

impl EpidemiologyMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Compute relative risk
    pub fn relative_risk<F>(
        &self,
        exposed_cases: F,
        exposed_total: F,
        unexposed_cases: F,
        unexposed_total: F,
    ) -> Result<F>
    where
        F: Float,
    {
        let risk_exposed = exposed_cases / exposed_total;
        let risk_unexposed = unexposed_cases / unexposed_total;

        if risk_unexposed > F::zero() {
            Ok(risk_exposed / risk_unexposed)
        } else {
            Err(MetricsError::InvalidInput(
                "Risk in unexposed group cannot be zero".to_string(),
            ))
        }
    }

    /// Compute odds ratio
    pub fn odds_ratio<F>(
        &self,
        exposed_cases: F,
        exposed_controls: F,
        unexposed_cases: F,
        unexposed_controls: F,
    ) -> Result<F>
    where
        F: Float,
    {
        if unexposed_cases > F::zero() && exposed_controls > F::zero() {
            Ok((exposed_cases * unexposed_controls) / (unexposed_cases * exposed_controls))
        } else {
            Err(MetricsError::InvalidInput(
                "Cannot compute odds ratio with zero values".to_string(),
            ))
        }
    }

    /// Compute population attributable risk
    pub fn population_attributable_risk<F>(
        &self,
        population_incidence: F,
        unexposed_incidence: F,
    ) -> Result<F>
    where
        F: Float,
    {
        Ok(population_incidence - unexposed_incidence)
    }
}

impl Default for EpidemiologyMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Survival analysis metrics
#[derive(Debug, Clone)]
pub struct SurvivalAnalysisMetrics;

impl SurvivalAnalysisMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Compute concordance index (C-index) for survival analysis
    pub fn concordance_index<F>(
        &self,
        risk_scores: &Array1<F>,
        survival_times: &Array1<F>,
        event_indicators: &Array1<bool>,
    ) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + std::iter::Sum,
    {
        if risk_scores.len() != survival_times.len()
            || survival_times.len() != event_indicators.len()
        {
            return Err(MetricsError::InvalidInput(
                "All arrays must have the same length".to_string(),
            ));
        }

        let mut concordant = 0;
        let mut total = 0;

        for i in 0..risk_scores.len() {
            if !event_indicators[i] {
                continue; // Skip censored observations
            }

            for j in 0..risk_scores.len() {
                if i == j {
                    continue;
                }

                // Compare pairs where one has an event
                if !event_indicators[j] && survival_times[j] > survival_times[i] {
                    // Censored observation with longer survival time
                    if risk_scores[i] > risk_scores[j] {
                        concordant += 1;
                    }
                    total += 1;
                } else if event_indicators[j] && survival_times[i] != survival_times[j] {
                    // Both have events at different _times
                    if (survival_times[i] < survival_times[j] && risk_scores[i] > risk_scores[j])
                        || (survival_times[i] > survival_times[j]
                            && risk_scores[i] < risk_scores[j])
                    {
                        concordant += 1;
                    }
                    total += 1;
                }
            }
        }

        if total > 0 {
            Ok(F::from(concordant).unwrap() / F::from(total).unwrap())
        } else {
            Ok(F::from(0.5).unwrap())
        }
    }
}

impl Default for SurvivalAnalysisMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Biomarker evaluation metrics
#[derive(Debug, Clone)]
pub struct BiomarkerMetrics;

impl BiomarkerMetrics {
    pub fn new() -> Self {
        Self
    }

    /// Compute biomarker discrimination metrics
    pub fn discrimination_metrics<F>(
        &self,
        biomarker_values: &Array1<F>,
        disease_status: &Array1<bool>,
    ) -> Result<BiomarkerDiscriminationMetrics<F>>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        if biomarker_values.len() != disease_status.len() {
            return Err(MetricsError::InvalidInput(
                "Biomarker _values and disease _status must have same length".to_string(),
            ));
        }

        // Compute AUC using trapezoidal rule
        let auc = self.compute_auc(biomarker_values, disease_status)?;

        // Find optimal threshold using Youden's J statistic
        let optimal_threshold = self.find_optimal_threshold(biomarker_values, disease_status)?;

        // Compute sensitivity and specificity at optimal threshold
        let (sensitivity, specificity) = self.compute_sensitivity_specificity(
            biomarker_values,
            disease_status,
            optimal_threshold,
        )?;

        Ok(BiomarkerDiscriminationMetrics {
            auc,
            sensitivity,
            specificity,
            optimal_threshold,
            youden_j: sensitivity + specificity - F::one(),
        })
    }

    /// Helper method to compute AUC
    fn compute_auc<F>(&self, values: &Array1<F>, labels: &Array1<bool>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        // Create (value, label) pairs and sort by value
        let mut pairs: Vec<(F, bool)> = values
            .iter()
            .zip(labels.iter())
            .map(|(&v, &l)| (v, l))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut auc = F::zero();
        let mut tp = F::zero();
        let mut fp = F::zero();

        let total_positives = F::from(labels.iter().filter(|&&x| x).count()).unwrap();
        let total_negatives = F::from(labels.iter().filter(|&&x| !x).count()).unwrap();

        for (_, label) in pairs {
            if label {
                tp = tp + F::one();
            } else {
                fp = fp + F::one();
                auc = auc + tp;
            }
        }

        if total_positives > F::zero() && total_negatives > F::zero() {
            Ok(auc / (total_positives * total_negatives))
        } else {
            Ok(F::from(0.5).unwrap())
        }
    }

    /// Helper method to find optimal threshold
    fn find_optimal_threshold<F>(&self, values: &Array1<F>, labels: &Array1<bool>) -> Result<F>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        let mut unique_values: Vec<F> = values.iter().cloned().collect();
        unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_values.dedup();

        let mut best_j = F::zero();
        let mut best_threshold = unique_values[0];

        for &threshold in &unique_values {
            let (sensitivity, specificity) =
                self.compute_sensitivity_specificity(values, labels, threshold)?;
            let j = sensitivity + specificity - F::one();

            if j > best_j {
                best_j = j;
                best_threshold = threshold;
            }
        }

        Ok(best_threshold)
    }

    /// Helper method to compute sensitivity and specificity
    fn compute_sensitivity_specificity<F>(
        &self,
        values: &Array1<F>,
        labels: &Array1<bool>,
        threshold: F,
    ) -> Result<(F, F)>
    where
        F: Float + scirs2_core::numeric::FromPrimitive + PartialOrd,
    {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (&value, &label) in values.iter().zip(labels.iter()) {
            let predicted_positive = value >= threshold;

            match (predicted_positive, label) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_count += 1,
                (false, false) => tn += 1,
            }
        }

        let sensitivity = if tp + fn_count > 0 {
            F::from(tp).unwrap() / F::from(tp + fn_count).unwrap()
        } else {
            F::zero()
        };

        let specificity = if tn + fp > 0 {
            F::from(tn).unwrap() / F::from(tn + fp).unwrap()
        } else {
            F::zero()
        };

        Ok((sensitivity, specificity))
    }
}

impl Default for BiomarkerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Result structures for biomedical metrics

/// Clinical trial efficacy results
#[derive(Debug, Clone)]
pub struct ClinicalEfficacyResult<F> {
    pub effect_size: F,
    pub confidence_interval_lower: F,
    pub confidence_interval_upper: F,
    pub p_value: F,
    pub treatment_mean: F,
    pub control_mean: F,
    pub outcome_type: EfficacyOutcome,
}

/// Drug-target interaction metrics
#[derive(Debug, Clone)]
pub struct DrugTargetMetrics<F> {
    pub correlation: F,
    pub rmse: F,
    pub mae: F,
    pub r_squared: F,
    pub concordance_index: F,
}

/// Genomic accuracy metrics
#[derive(Debug, Clone)]
pub struct GenomicAccuracyMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

/// Biomarker discrimination metrics
#[derive(Debug, Clone)]
pub struct BiomarkerDiscriminationMetrics<F> {
    pub auc: F,
    pub sensitivity: F,
    pub specificity: F,
    pub optimal_threshold: F,
    pub youden_j: F,
}

/// Types of clinical efficacy outcomes
#[derive(Debug, Clone, Copy)]
pub enum EfficacyOutcome {
    Continuous,
    Binary,
    TimeToEvent,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_biomedical_suite_creation() {
        let suite = BiomedicalSuite::new();
        assert_eq!(suite.domain_name(), "Biomedical & Healthcare");

        let metrics = suite.available_metrics();
        assert!(metrics.contains(&"sensitivity"));
        assert!(metrics.contains(&"specificity"));
        assert!(metrics.contains(&"dice_coefficient"));
    }

    #[test]
    fn test_dice_coefficient() {
        let imaging = MedicalImagingMetrics::new();
        let predicted = array![[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let ground_truth = array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];

        let dice = imaging.dice_coefficient(&predicted, &ground_truth).unwrap();
        assert!(dice > 0.0 && dice <= 1.0);
    }

    #[test]
    fn test_relative_risk() {
        let epi = EpidemiologyMetrics::new();
        let rr = epi.relative_risk(10.0, 100.0, 5.0, 100.0).unwrap();
        assert_eq!(rr, 2.0);
    }

    #[test]
    fn test_number_needed_to_treat() {
        let clinical = ClinicalTrialMetrics::new();
        let nnt = clinical.number_needed_to_treat(0.8, 0.6).unwrap();
        assert!((nnt - 5.0).abs() < 1e-10, "Expected NNT ~5.0, got {}", nnt);
    }

    #[test]
    fn test_tanimoto_similarity() {
        let drug_discovery = DrugDiscoveryMetrics::new();
        let fp1 = array![1, 1, 0, 1, 0];
        let fp2 = array![1, 0, 0, 1, 1];

        let similarity = drug_discovery.tanimoto_similarity(&fp1, &fp2).unwrap();
        assert!((0.0..=1.0).contains(&similarity));
    }
}
