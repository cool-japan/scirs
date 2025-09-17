//! Molecular graph evaluation metrics for Graph Neural Networks
//!
//! This module provides metrics for evaluating molecular property prediction,
//! drug discovery, and chemical analysis tasks using graph neural networks.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::MolecularStructure;
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Molecular graph evaluation metrics
#[derive(Debug, Clone)]
pub struct MolecularGraphMetrics {
    /// Molecular property prediction metrics
    pub property_prediction: MolecularPropertyMetrics,
    /// Drug discovery evaluation metrics
    pub drug_discovery: DrugDiscoveryMetrics,
    /// Toxicity prediction metrics
    pub toxicity_metrics: ToxicityMetrics,
    /// Drug-target interaction metrics
    pub dti_prediction: DtiPredictionMetrics,
    /// Chemical similarity metrics
    pub chemical_similarity: ChemicalSimilarityMetrics,
    /// Reaction prediction metrics
    pub reaction_prediction: ReactionPredictionMetrics,
}

/// Molecular property prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularPropertyMetrics {
    /// Per-property prediction performance
    pub property_metrics: HashMap<String, PropertyMetrics>,
    /// Overall regression performance
    pub overall_mae: f64,
    /// Overall regression R²
    pub overall_r2: f64,
    /// Classification accuracy (for binary properties)
    pub classification_accuracy: f64,
}

impl Default for MolecularPropertyMetrics {
    fn default() -> Self {
        Self {
            property_metrics: HashMap::new(),
            overall_mae: 0.0,
            overall_r2: 0.0,
            classification_accuracy: 0.0,
        }
    }
}

impl MolecularPropertyMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Individual molecular property metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// R-squared score
    pub r2_score: f64,
}

impl Default for PropertyMetrics {
    fn default() -> Self {
        Self {
            mae: 0.0,
            rmse: 0.0,
            r2_score: 0.0,
        }
    }
}

impl PropertyMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Drug discovery evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugDiscoveryMetrics {
    /// ADMET prediction accuracy
    pub admet_accuracy: f64,
    /// Bioactivity prediction AUC
    pub bioactivity_auc: f64,
    /// Lead optimization score
    pub lead_optimization_score: f64,
    /// Druglikeness prediction accuracy
    pub druglikeness_accuracy: f64,
}

impl Default for DrugDiscoveryMetrics {
    fn default() -> Self {
        Self {
            admet_accuracy: 0.0,
            bioactivity_auc: 0.0,
            lead_optimization_score: 0.0,
            druglikeness_accuracy: 0.0,
        }
    }
}

impl DrugDiscoveryMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Toxicity prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityMetrics {
    /// Acute toxicity prediction accuracy
    pub acute_toxicity_accuracy: f64,
    /// Chronic toxicity prediction accuracy
    pub chronic_toxicity_accuracy: f64,
    /// Carcinogenicity prediction AUC
    pub carcinogenicity_auc: f64,
    /// Mutagenicity prediction AUC
    pub mutagenicity_auc: f64,
}

impl Default for ToxicityMetrics {
    fn default() -> Self {
        Self {
            acute_toxicity_accuracy: 0.0,
            chronic_toxicity_accuracy: 0.0,
            carcinogenicity_auc: 0.0,
            mutagenicity_auc: 0.0,
        }
    }
}

impl ToxicityMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Drug-target interaction prediction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtiPredictionMetrics {
    /// DTI prediction AUC
    pub dti_auc: f64,
    /// DTI prediction precision
    pub dti_precision: f64,
    /// DTI prediction recall
    pub dti_recall: f64,
    /// Binding affinity prediction MAE
    pub affinity_mae: f64,
}

impl Default for DtiPredictionMetrics {
    fn default() -> Self {
        Self {
            dti_auc: 0.0,
            dti_precision: 0.0,
            dti_recall: 0.0,
            affinity_mae: 0.0,
        }
    }
}

impl DtiPredictionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Chemical similarity evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemicalSimilarityMetrics {
    /// Tanimoto similarity correlation
    pub tanimoto_correlation: f64,
    /// Morgan fingerprint similarity correlation
    pub morgan_correlation: f64,
    /// Molecular descriptor similarity correlation
    pub descriptor_correlation: f64,
}

impl Default for ChemicalSimilarityMetrics {
    fn default() -> Self {
        Self {
            tanimoto_correlation: 0.0,
            morgan_correlation: 0.0,
            descriptor_correlation: 0.0,
        }
    }
}

impl ChemicalSimilarityMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Reaction prediction evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionPredictionMetrics {
    /// Reaction product prediction accuracy
    pub product_accuracy: f64,
    /// Reaction condition prediction accuracy
    pub condition_accuracy: f64,
    /// Yield prediction MAE
    pub yield_mae: f64,
}

impl Default for ReactionPredictionMetrics {
    fn default() -> Self {
        Self {
            product_accuracy: 0.0,
            condition_accuracy: 0.0,
            yield_mae: 0.0,
        }
    }
}

impl ReactionPredictionMetrics {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MolecularGraphMetrics {
    pub fn new() -> Self {
        Self {
            property_prediction: MolecularPropertyMetrics::new(),
            drug_discovery: DrugDiscoveryMetrics::new(),
            toxicity_metrics: ToxicityMetrics::new(),
            dti_prediction: DtiPredictionMetrics::new(),
            chemical_similarity: ChemicalSimilarityMetrics::new(),
            reaction_prediction: ReactionPredictionMetrics::new(),
        }
    }

    /// Evaluate molecular property prediction
    pub fn evaluate_molecular_properties(
        &mut self,
        predictions: &HashMap<String, Vec<f64>>, // property_name -> predictions
        ground_truth: &HashMap<String, Vec<f64>>, // property_name -> true_values
    ) -> Result<()> {
        for (property, pred_values) in predictions {
            if let Some(true_values) = ground_truth.get(property) {
                if pred_values.len() != true_values.len() {
                    continue;
                }

                let mut property_metrics = PropertyMetrics::new();

                // Calculate MAE
                let mae: f64 = pred_values
                    .iter()
                    .zip(true_values.iter())
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>()
                    / pred_values.len() as f64;

                // Calculate RMSE
                let mse: f64 = pred_values
                    .iter()
                    .zip(true_values.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / pred_values.len() as f64;
                let rmse = mse.sqrt();

                // Calculate R²
                let true_mean = true_values.iter().sum::<f64>() / true_values.len() as f64;
                let ss_tot: f64 = true_values.iter().map(|t| (t - true_mean).powi(2)).sum();
                let ss_res: f64 = pred_values
                    .iter()
                    .zip(true_values.iter())
                    .map(|(p, t)| (t - p).powi(2))
                    .sum();

                let r2 = if ss_tot > 0.0 {
                    1.0 - ss_res / ss_tot
                } else {
                    0.0
                };

                property_metrics.mae = mae;
                property_metrics.rmse = rmse;
                property_metrics.r2_score = r2;

                self.property_prediction
                    .property_metrics
                    .insert(property.clone(), property_metrics);
            }
        }

        // Calculate overall metrics
        if !self.property_prediction.property_metrics.is_empty() {
            self.property_prediction.overall_mae = self
                .property_prediction
                .property_metrics
                .values()
                .map(|m| m.mae)
                .sum::<f64>()
                / self.property_prediction.property_metrics.len() as f64;

            self.property_prediction.overall_r2 = self
                .property_prediction
                .property_metrics
                .values()
                .map(|m| m.r2_score)
                .sum::<f64>()
                / self.property_prediction.property_metrics.len() as f64;
        }

        Ok(())
    }

    /// Evaluate drug discovery metrics
    pub fn evaluate_drug_discovery(
        &mut self,
        admet_predictions: &[bool],
        admet_ground_truth: &[bool],
        bioactivity_scores: &[f64],
        bioactivity_labels: &[bool],
    ) -> Result<()> {
        // ADMET accuracy
        if admet_predictions.len() == admet_ground_truth.len() && !admet_predictions.is_empty() {
            let correct = admet_predictions
                .iter()
                .zip(admet_ground_truth.iter())
                .filter(|(p, t)| p == t)
                .count();
            self.drug_discovery.admet_accuracy = correct as f64 / admet_predictions.len() as f64;
        }

        // Bioactivity AUC (simplified calculation)
        if bioactivity_scores.len() == bioactivity_labels.len() && !bioactivity_scores.is_empty() {
            self.drug_discovery.bioactivity_auc =
                self.calculate_auc(bioactivity_scores, bioactivity_labels);
        }

        Ok(())
    }

    fn calculate_auc(&self, scores: &[f64], labels: &[bool]) -> f64 {
        let mut score_label_pairs: Vec<(f64, bool)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();

        score_label_pairs
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = labels.iter().filter(|&&l| l).count() as f64;
        let total_negatives = labels.iter().filter(|&&l| !l).count() as f64;

        if total_positives == 0.0 || total_negatives == 0.0 {
            return 0.5;
        }

        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_fpr = 0.0;

        for (_, is_positive) in score_label_pairs {
            if is_positive {
                tp += 1.0;
            } else {
                fp += 1.0;
                let tpr = tp / total_positives;
                let fpr = fp / total_negatives;
                auc += tpr * (fpr - prev_fpr);
                prev_fpr = fpr;
            }
        }

        auc
    }
}

impl Default for MolecularGraphMetrics {
    fn default() -> Self {
        Self::new()
    }
}
