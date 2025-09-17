//! Speaker identification, verification, and diarization metrics
//!
//! This module provides comprehensive metrics for speaker-related audio processing tasks,
//! including speaker identification, verification, and diarization evaluation.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Speaker identification and verification metrics
#[derive(Debug, Clone)]
pub struct SpeakerMetrics {
    /// Speaker identification metrics
    pub identification: SpeakerIdentificationMetrics,
    /// Speaker verification metrics
    pub verification: SpeakerVerificationMetrics,
    /// Speaker diarization metrics
    pub diarization: SpeakerDiarizationMetrics,
}

/// Speaker identification metrics
#[derive(Debug, Clone)]
pub struct SpeakerIdentificationMetrics {
    /// Top-1 accuracy
    top1_accuracy: f64,
    /// Top-5 accuracy
    top5_accuracy: f64,
    /// Rank-based metrics
    mean_reciprocal_rank: f64,
    /// Confusion matrix for speakers
    speaker_confusion: HashMap<(String, String), usize>,
}

/// Speaker verification metrics
#[derive(Debug, Clone)]
pub struct SpeakerVerificationMetrics {
    /// Equal Error Rate (EER)
    eer: f64,
    /// Detection Cost Function (DCF)
    dcf: f64,
    /// Minimum DCF
    min_dcf: f64,
    /// Area Under Curve (AUC)
    auc: f64,
    /// False Acceptance Rate at specific threshold
    far_at_threshold: HashMap<f64, f64>,
    /// False Rejection Rate at specific threshold
    frr_at_threshold: HashMap<f64, f64>,
}

/// Speaker diarization metrics
#[derive(Debug, Clone, Default)]
pub struct SpeakerDiarizationMetrics {
    /// Diarization Error Rate (DER)
    der: f64,
    /// Jaccard Error Rate (JER)
    jer: f64,
    /// Speaker confusion error
    speaker_confusion_error: f64,
    /// False alarm error
    false_alarm_error: f64,
    /// Missed speech error
    missed_speech_error: f64,
}

/// Speaker recognition results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerResults {
    /// Speaker identification accuracy
    pub identification_accuracy: Option<f64>,
    /// Speaker verification EER
    pub verification_eer: Option<f64>,
    /// Diarization Error Rate
    pub diarization_der: Option<f64>,
}

impl SpeakerMetrics {
    /// Create new speaker metrics
    pub fn new() -> Self {
        Self {
            identification: SpeakerIdentificationMetrics::new(),
            verification: SpeakerVerificationMetrics::new(),
            diarization: SpeakerDiarizationMetrics::new(),
        }
    }

    /// Evaluate speaker identification performance
    pub fn evaluate_identification(
        &mut self,
        true_speakers: &[String],
        predicted_speakers: &[String],
        prediction_scores: Option<&[Vec<f64>]>,
    ) -> Result<f64> {
        if true_speakers.len() != predicted_speakers.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched array lengths for speaker identification".to_string(),
            ));
        }

        // Calculate top-1 accuracy
        let correct = true_speakers
            .iter()
            .zip(predicted_speakers.iter())
            .filter(|(true_spk, pred_spk)| true_spk == pred_spk)
            .count();

        let accuracy = correct as f64 / true_speakers.len() as f64;
        self.identification.top1_accuracy = accuracy;

        // Calculate top-5 accuracy if scores provided
        if let Some(scores) = prediction_scores {
            let top5_correct = true_speakers
                .iter()
                .zip(scores.iter())
                .filter(|(true_spk, pred_scores)| {
                    // Get top 5 predictions
                    let mut indexed_scores: Vec<(usize, f64)> = pred_scores
                        .iter()
                        .enumerate()
                        .map(|(i, &score)| (i, score))
                        .collect();
                    indexed_scores
                        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    // Check if true speaker is in top 5
                    indexed_scores.iter().take(5).any(|(idx, _)| {
                        // This is simplified - would need actual speaker ID mapping
                        false // Placeholder
                    })
                })
                .count();

            self.identification.top5_accuracy = top5_correct as f64 / true_speakers.len() as f64;
        }

        // Update confusion matrix
        for (true_spk, pred_spk) in true_speakers.iter().zip(predicted_speakers.iter()) {
            let key = (true_spk.clone(), pred_spk.clone());
            *self
                .identification
                .speaker_confusion
                .entry(key)
                .or_insert(0) += 1;
        }

        Ok(accuracy)
    }

    /// Evaluate speaker verification performance
    pub fn evaluate_verification(
        &mut self,
        true_labels: &[bool],
        similarity_scores: &[f64],
    ) -> Result<f64> {
        if true_labels.len() != similarity_scores.len() {
            return Err(MetricsError::InvalidInput(
                "Mismatched array lengths for speaker verification".to_string(),
            ));
        }

        // Calculate EER
        let eer = self.calculate_eer(true_labels, similarity_scores)?;
        self.verification.eer = eer;

        // Calculate DCF
        let dcf = self.calculate_dcf(true_labels, similarity_scores)?;
        self.verification.dcf = dcf;

        Ok(eer)
    }

    /// Evaluate speaker diarization performance
    pub fn evaluate_diarization(
        &mut self,
        reference_segments: &[(f64, f64, String)], // (start, end, speaker_id)
        hypothesis_segments: &[(f64, f64, String)],
    ) -> Result<f64> {
        // Calculate Diarization Error Rate (DER)
        let der = self.calculate_der(reference_segments, hypothesis_segments)?;
        self.diarization.der = der;

        // Calculate Jaccard Error Rate (JER)
        let jer = self.calculate_jer(reference_segments, hypothesis_segments)?;
        self.diarization.jer = jer;

        Ok(der)
    }

    /// Calculate Equal Error Rate for speaker verification
    fn calculate_eer(&self, true_labels: &[bool], scores: &[f64]) -> Result<f64> {
        // Create pairs and sort by score
        let mut score_label_pairs: Vec<(f64, bool)> = scores
            .iter()
            .zip(true_labels.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = true_labels.iter().filter(|&&x| x).count() as f64;
        let total_negatives = true_labels.iter().filter(|&&x| !x).count() as f64;

        if total_positives == 0.0 || total_negatives == 0.0 {
            return Ok(0.0);
        }

        let mut min_diff = f64::INFINITY;
        let mut best_eer = 0.0;
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (_, label) in score_label_pairs.iter().rev() {
            if *label {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let tpr = true_positives / total_positives;
            let fpr = false_positives / total_negatives;
            let fnr = 1.0 - tpr;

            let diff = (fpr - fnr).abs();
            if diff < min_diff {
                min_diff = diff;
                best_eer = (fpr + fnr) / 2.0;
            }
        }

        Ok(best_eer)
    }

    /// Calculate Detection Cost Function for speaker verification
    fn calculate_dcf(&self, true_labels: &[bool], scores: &[f64]) -> Result<f64> {
        // DCF parameters (NIST SRE standard)
        let c_miss = 1.0;
        let c_fa = 1.0;
        let p_target = 0.01;

        let mut score_label_pairs: Vec<(f64, bool)> = scores
            .iter()
            .zip(true_labels.iter())
            .map(|(&score, &label)| (score, label))
            .collect();

        score_label_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives = true_labels.iter().filter(|&&x| x).count() as f64;
        let total_negatives = true_labels.iter().filter(|&&x| !x).count() as f64;

        let mut min_dcf = f64::INFINITY;
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (_, label) in score_label_pairs.iter().rev() {
            if *label {
                true_positives += 1.0;
            } else {
                false_positives += 1.0;
            }

            let pmiss = 1.0 - (true_positives / total_positives);
            let pfa = false_positives / total_negatives;

            let dcf = c_miss * pmiss * p_target + c_fa * pfa * (1.0 - p_target);
            min_dcf = min_dcf.min(dcf);
        }

        Ok(min_dcf)
    }

    /// Calculate Diarization Error Rate
    fn calculate_der(
        &self,
        reference: &[(f64, f64, String)],
        hypothesis: &[(f64, f64, String)],
    ) -> Result<f64> {
        // Find total duration
        let max_time = reference
            .iter()
            .chain(hypothesis.iter())
            .map(|(_, end, _)| *end)
            .fold(0.0f64, f64::max);

        if max_time <= 0.0 {
            return Ok(0.0);
        }

        // Time resolution for frame-based evaluation
        let resolution = 0.01; // 10ms
        let num_frames = (max_time / resolution).ceil() as usize;

        // Create frame-level representations
        let mut ref_frames = vec![String::new(); num_frames];
        let mut hyp_frames = vec![String::new(); num_frames];

        // Fill reference frames
        for (start, end, speaker) in reference {
            let start_frame = (*start / resolution) as usize;
            let end_frame = ((*end / resolution) as usize).min(num_frames);
            for i in start_frame..end_frame {
                ref_frames[i] = speaker.clone();
            }
        }

        // Fill hypothesis frames
        for (start, end, speaker) in hypothesis {
            let start_frame = (*start / resolution) as usize;
            let end_frame = ((*end / resolution) as usize).min(num_frames);
            for i in start_frame..end_frame {
                hyp_frames[i] = speaker.clone();
            }
        }

        // Calculate frame-level errors
        let mut errors = 0;
        let mut total_speech_frames = 0;

        for (ref_speaker, hyp_speaker) in ref_frames.iter().zip(hyp_frames.iter()) {
            if !ref_speaker.is_empty() {
                total_speech_frames += 1;
                if ref_speaker != hyp_speaker {
                    errors += 1;
                }
            }
        }

        if total_speech_frames == 0 {
            Ok(0.0)
        } else {
            Ok(errors as f64 / total_speech_frames as f64)
        }
    }

    /// Calculate Jaccard Error Rate
    fn calculate_jer(
        &self,
        reference: &[(f64, f64, String)],
        hypothesis: &[(f64, f64, String)],
    ) -> Result<f64> {
        // Simplified JER calculation
        // In practice, this would involve more complex speaker cluster matching
        let ref_speakers: std::collections::HashSet<_> =
            reference.iter().map(|(_, _, speaker)| speaker).collect();
        let hyp_speakers: std::collections::HashSet<_> =
            hypothesis.iter().map(|(_, _, speaker)| speaker).collect();

        let intersection = ref_speakers.intersection(&hyp_speakers).count();
        let union = ref_speakers.union(&hyp_speakers).count();

        if union == 0 {
            Ok(0.0)
        } else {
            Ok(1.0 - (intersection as f64 / union as f64))
        }
    }

    /// Get speaker identification accuracy
    pub fn get_identification_accuracy(&self) -> f64 {
        self.identification.top1_accuracy
    }

    /// Get speaker verification EER
    pub fn get_verification_eer(&self) -> f64 {
        self.verification.eer
    }

    /// Get diarization error rate
    pub fn get_diarization_der(&self) -> f64 {
        self.diarization.der
    }
}

impl SpeakerIdentificationMetrics {
    /// Create new speaker identification metrics
    pub fn new() -> Self {
        Self {
            top1_accuracy: 0.0,
            top5_accuracy: 0.0,
            mean_reciprocal_rank: 0.0,
            speaker_confusion: HashMap::new(),
        }
    }

    /// Get top-1 accuracy
    pub fn get_top1_accuracy(&self) -> f64 {
        self.top1_accuracy
    }

    /// Get top-5 accuracy
    pub fn get_top5_accuracy(&self) -> f64 {
        self.top5_accuracy
    }

    /// Get confusion matrix
    pub fn get_confusion_matrix(&self) -> &HashMap<(String, String), usize> {
        &self.speaker_confusion
    }
}

impl SpeakerVerificationMetrics {
    /// Create new speaker verification metrics
    pub fn new() -> Self {
        Self {
            eer: 0.0,
            dcf: 0.0,
            min_dcf: 0.0,
            auc: 0.0,
            far_at_threshold: HashMap::new(),
            frr_at_threshold: HashMap::new(),
        }
    }

    /// Get Equal Error Rate
    pub fn get_eer(&self) -> f64 {
        self.eer
    }

    /// Get Detection Cost Function
    pub fn get_dcf(&self) -> f64 {
        self.dcf
    }

    /// Get minimum DCF
    pub fn get_min_dcf(&self) -> f64 {
        self.min_dcf
    }
}

impl SpeakerDiarizationMetrics {
    /// Create new speaker diarization metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Get Diarization Error Rate
    pub fn get_der(&self) -> f64 {
        self.der
    }

    /// Get Jaccard Error Rate
    pub fn get_jer(&self) -> f64 {
        self.jer
    }

    /// Get speaker confusion error
    pub fn get_speaker_confusion_error(&self) -> f64 {
        self.speaker_confusion_error
    }
}

impl Default for SpeakerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SpeakerIdentificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SpeakerVerificationMetrics {
    fn default() -> Self {
        Self::new()
    }
}
