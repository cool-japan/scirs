//! Audio processing and speech recognition metrics
//!
//! This module provides specialized metrics for audio processing tasks including:
//! - Speech recognition (ASR) evaluation
//! - Audio classification metrics
//! - Music information retrieval (MIR) metrics
//! - Audio quality assessment
//! - Sound event detection metrics
//! - Speaker identification and verification
//! - Audio similarity and retrieval metrics

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::{DomainEvaluationResult, DomainMetrics};
use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// Re-export all submodules
pub mod audio_classification;
pub mod audio_quality;
pub mod audio_similarity;
pub mod music_information;
pub mod sound_event_detection;
pub mod speaker_metrics;
pub mod speech_recognition;

// Re-export key types for backward compatibility
pub use speech_recognition::{
    BleuCalculator, BleuSmoothing, CerCalculator, ConfidenceMetrics, PerCalculator,
    SpeechRecognitionMetrics, SpeechRecognitionResults, WerCalculator,
};

pub use audio_classification::{
    AudioClassificationMetrics, AudioClassificationResults, AudioSpecificMetrics,
    BoundaryDetectionMetrics, TemporalAudioMetrics,
};

pub use music_information::{
    BeatTrackingMetrics, ChordRecognitionMetrics, ContinuityMetrics, CoverSongMetrics,
    KeyDetectionMetrics, MusicInformationMetrics, MusicInformationResults, MusicSimilarityMetrics,
    TempoEstimationMetrics,
};

pub use audio_quality::{
    AudioQualityMetrics, AudioQualityResults, IntelligibilityMetrics, ObjectiveAudioMetrics,
    PerceptualAudioMetrics, SpectralDistortionMetrics,
};

pub use sound_event_detection::{
    ClassWiseEventMetrics, EventBasedMetrics, SegmentBasedMetrics, SoundEvent,
    SoundEventDetectionMetrics, SoundEventResults,
};

pub use speaker_metrics::{
    SpeakerDiarizationMetrics, SpeakerIdentificationMetrics, SpeakerMetrics, SpeakerResults,
    SpeakerVerificationMetrics,
};

pub use audio_similarity::{
    AcousticSimilarityMetrics, AudioSimilarityMetrics, AudioSimilarityResults,
    ContentBasedRetrievalMetrics, SemanticSimilarityMetrics,
};

/// Comprehensive audio processing metrics suite
#[derive(Debug)]
pub struct AudioProcessingMetrics {
    /// Speech recognition metrics
    pub speech_recognition: SpeechRecognitionMetrics,
    /// Audio classification metrics
    pub audio_classification: AudioClassificationMetrics,
    /// Music information retrieval metrics
    pub music_metrics: MusicInformationMetrics,
    /// Audio quality metrics
    pub quality_metrics: AudioQualityMetrics,
    /// Sound event detection metrics
    pub event_detection: SoundEventDetectionMetrics,
    /// Speaker metrics
    pub speaker_metrics: SpeakerMetrics,
    /// Audio similarity metrics
    pub similarity_metrics: AudioSimilarityMetrics,
}

/// Audio evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEvaluationResults {
    /// Speech recognition results
    pub speech_recognition: Option<SpeechRecognitionResults>,
    /// Audio classification results
    pub audio_classification: Option<AudioClassificationResults>,
    /// Music information retrieval results
    pub music_information: Option<MusicInformationResults>,
    /// Audio quality results
    pub quality_assessment: Option<AudioQualityResults>,
    /// Sound event detection results
    pub event_detection: Option<SoundEventResults>,
    /// Speaker recognition results
    pub speaker_recognition: Option<SpeakerResults>,
    /// Audio similarity results
    pub similarity: Option<AudioSimilarityResults>,
}

/// Comprehensive audio evaluation report
#[derive(Debug)]
pub struct AudioEvaluationReport {
    /// Executive summary
    pub summary: AudioSummary,
    /// Detailed results by domain
    pub detailed_results: AudioEvaluationResults,
    /// Performance insights
    pub insights: Vec<AudioInsight>,
    /// Recommendations
    pub recommendations: Vec<AudioRecommendation>,
}

/// Audio evaluation summary
#[derive(Debug)]
pub struct AudioSummary {
    /// Overall performance score
    pub overall_score: f64,
    /// Best performing domain
    pub best_domain: String,
    /// Worst performing domain
    pub worst_domain: String,
    /// Key strengths
    pub strengths: Vec<String>,
    /// Areas for improvement
    pub improvements: Vec<String>,
}

/// Audio performance insight
#[derive(Debug)]
pub struct AudioInsight {
    /// Insight category
    pub category: AudioInsightCategory,
    /// Insight title
    pub title: String,
    /// Insight description
    pub description: String,
    /// Supporting metrics
    pub metrics: HashMap<String, f64>,
}

/// Audio insight categories
#[derive(Debug)]
pub enum AudioInsightCategory {
    Performance,
    Quality,
    Robustness,
    Efficiency,
    UserExperience,
}

/// Audio improvement recommendation
#[derive(Debug)]
pub struct AudioRecommendation {
    /// Recommendation priority
    pub priority: RecommendationPriority,
    /// Recommendation title
    pub title: String,
    /// Recommendation description
    pub description: String,
    /// Expected impact
    pub expected_impact: f64,
    /// Implementation effort
    pub implementation_effort: ImplementationEffort,
}

/// Recommendation priority levels
#[derive(Debug)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort levels
#[derive(Debug)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

impl AudioProcessingMetrics {
    /// Create new audio processing metrics suite
    pub fn new() -> Self {
        Self {
            speech_recognition: SpeechRecognitionMetrics::new(),
            audio_classification: AudioClassificationMetrics::new(),
            music_metrics: MusicInformationMetrics::new(),
            quality_metrics: AudioQualityMetrics::new(),
            event_detection: SoundEventDetectionMetrics::new(),
            speaker_metrics: SpeakerMetrics::new(),
            similarity_metrics: AudioSimilarityMetrics::new(),
        }
    }

    /// Evaluate speech recognition output
    pub fn evaluate_speech_recognition(
        &mut self,
        reference_text: &[String],
        hypothesis_text: &[String],
        reference_phones: Option<&[Vec<String>]>,
        hypothesis_phones: Option<&[Vec<String>]>,
        confidence_scores: Option<&[f64]>,
    ) -> Result<SpeechRecognitionResults> {
        self.speech_recognition.evaluate_recognition(
            reference_text,
            hypothesis_text,
            reference_phones,
            hypothesis_phones,
            confidence_scores,
        )
    }

    /// Evaluate audio classification performance
    pub fn evaluate_audio_classification<F>(
        &mut self,
        y_true: ArrayView1<i32>,
        y_pred: ArrayView1<i32>,
        y_scores: Option<ArrayView2<F>>,
        frame_predictions: Option<ArrayView2<i32>>,
    ) -> Result<AudioClassificationResults>
    where
        F: Float + std::fmt::Debug,
    {
        self.audio_classification
            .compute_metrics(y_true, y_pred, y_scores, frame_predictions)
    }

    /// Evaluate music information retrieval tasks
    pub fn evaluate_music_information(
        &mut self,
        beat_annotations: Option<(&[f64], &[f64])>, // (reference_beats, estimated_beats)
        chord_annotations: Option<(&[String], &[String])>, // (reference_chords, estimated_chords)
        key_annotations: Option<(String, String)>,  // (reference_key, estimated_key)
        tempo_annotations: Option<(f64, f64)>,      // (reference_tempo, estimated_tempo)
    ) -> Result<MusicInformationResults> {
        let mut results = MusicInformationResults {
            beat_f_measure: None,
            chord_accuracy: None,
            key_accuracy: None,
            tempo_accuracy: None,
            similarity_map: None,
        };

        if let Some((ref_beats, est_beats)) = beat_annotations {
            let f_measure = self
                .music_metrics
                .evaluate_beats(ref_beats, est_beats, 0.07)?;
            results.beat_f_measure = Some(f_measure);
        }

        if let Some((ref_chords, est_chords)) = chord_annotations {
            let accuracy = self.music_metrics.evaluate_chords(ref_chords, est_chords)?;
            results.chord_accuracy = Some(accuracy);
        }

        if let Some((ref_key, est_key)) = key_annotations {
            let accuracy = if ref_key == est_key { 1.0 } else { 0.0 };
            results.key_accuracy = Some(accuracy);
        }

        if let Some((ref_tempo, est_tempo)) = tempo_annotations {
            let accuracy = self
                .music_metrics
                .evaluate_tempo(ref_tempo, est_tempo, 0.04)?;
            results.tempo_accuracy = Some(accuracy);
        }

        Ok(results)
    }

    /// Evaluate audio quality
    pub fn evaluate_audio_quality<F>(
        &mut self,
        reference_audio: ArrayView1<F>,
        degraded_audio: ArrayView1<F>,
        sample_rate: f64,
    ) -> Result<AudioQualityResults>
    where
        F: Float + std::fmt::Debug + std::iter::Sum,
    {
        self.quality_metrics
            .evaluate_quality(reference_audio, degraded_audio, sample_rate)
    }

    /// Evaluate sound event detection
    pub fn evaluate_sound_event_detection(
        &mut self,
        reference_events: &[SoundEvent],
        predicted_events: &[SoundEvent],
        tolerance: f64,
    ) -> Result<SoundEventResults> {
        self.event_detection
            .evaluate_events(reference_events, predicted_events, tolerance)
    }

    /// Evaluate speaker recognition tasks
    pub fn evaluate_speaker_recognition(
        &mut self,
        identification_data: Option<(&[String], &[String])>, // (true_speakers, predicted_speakers)
        verification_data: Option<(&[bool], &[f64])>,        // (true_labels, similarity_scores)
        diarization_data: Option<(&[(f64, f64, String)], &[(f64, f64, String)])>, // (reference, hypothesis)
    ) -> Result<SpeakerResults> {
        let mut results = SpeakerResults {
            identification_accuracy: None,
            verification_eer: None,
            diarization_der: None,
        };

        if let Some((true_speakers, pred_speakers)) = identification_data {
            let accuracy =
                self.speaker_metrics
                    .evaluate_identification(true_speakers, pred_speakers, None)?;
            results.identification_accuracy = Some(accuracy);
        }

        if let Some((true_labels, scores)) = verification_data {
            let eer = self
                .speaker_metrics
                .evaluate_verification(true_labels, scores)?;
            results.verification_eer = Some(eer);
        }

        if let Some((reference, hypothesis)) = diarization_data {
            let der = self
                .speaker_metrics
                .evaluate_diarization(reference, hypothesis)?;
            results.diarization_der = Some(der);
        }

        Ok(results)
    }

    /// Evaluate audio similarity
    pub fn evaluate_audio_similarity<F>(
        &mut self,
        query_ids: &[String],
        relevant_docs: &HashMap<String, Vec<String>>,
        retrieved_docs: &HashMap<String, Vec<String>>,
        acoustic_features: Option<(
            &HashMap<String, ArrayView1<F>>,
            &HashMap<String, ArrayView1<F>>,
        )>,
        semantic_data: Option<(&HashMap<String, Vec<String>>, &HashMap<String, Vec<String>>)>,
    ) -> Result<AudioSimilarityResults>
    where
        F: Float + std::fmt::Debug,
    {
        // Evaluate content-based retrieval
        let k_values = vec![1, 5, 10, 20];
        self.similarity_metrics.evaluate_retrieval(
            query_ids,
            relevant_docs,
            retrieved_docs,
            &k_values,
        )?;

        // Evaluate acoustic similarity if features provided
        if let Some((ref_features, query_features)) = acoustic_features {
            self.similarity_metrics
                .evaluate_acoustic_similarity(ref_features, query_features)?;
        }

        // Evaluate semantic similarity if data provided
        if let Some((ref_tags, query_tags)) = semantic_data {
            self.similarity_metrics
                .evaluate_semantic_similarity(ref_tags, query_tags, None, None)?;
        }

        Ok(self.similarity_metrics.get_results())
    }

    /// Create comprehensive audio evaluation report
    pub fn create_comprehensive_report(
        &self,
        results: &AudioEvaluationResults,
    ) -> AudioEvaluationReport {
        AudioEvaluationReport::new(results)
    }
}

impl AudioEvaluationReport {
    /// Create new audio evaluation report
    pub fn new(results: &AudioEvaluationResults) -> Self {
        let summary = AudioSummary {
            overall_score: 0.75,
            best_domain: "Speech Recognition".to_string(),
            worst_domain: "Music Information Retrieval".to_string(),
            strengths: vec![
                "High accuracy".to_string(),
                "Good temporal consistency".to_string(),
            ],
            improvements: vec!["Better chord recognition".to_string()],
        };

        Self {
            summary,
            detailed_results: results.clone(),
            insights: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    /// Add performance insight
    pub fn add_insight(&mut self, insight: AudioInsight) {
        self.insights.push(insight);
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: AudioRecommendation) {
        self.recommendations.push(recommendation);
    }

    /// Generate summary statistics
    pub fn generate_summary(&mut self) {
        // Update summary based on detailed results
        let mut domain_scores = Vec::new();

        if let Some(ref sr_results) = self.detailed_results.speech_recognition {
            domain_scores.push(("Speech Recognition", 1.0 - sr_results.wer));
        }

        if let Some(ref ac_results) = self.detailed_results.audio_classification {
            domain_scores.push(("Audio Classification", ac_results.accuracy));
        }

        if let Some(ref mi_results) = self.detailed_results.music_information {
            if let Some(beat_f1) = mi_results.beat_f_measure {
                domain_scores.push(("Music Information Retrieval", beat_f1));
            }
        }

        if let Some(ref aq_results) = self.detailed_results.quality_assessment {
            let normalized_snr = (aq_results.snr / 40.0).min(1.0).max(0.0);
            domain_scores.push(("Audio Quality", normalized_snr));
        }

        if !domain_scores.is_empty() {
            // Find best and worst domains
            domain_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            self.summary.best_domain = domain_scores.first().unwrap().0.to_string();
            self.summary.worst_domain = domain_scores.last().unwrap().0.to_string();

            // Calculate overall score
            self.summary.overall_score = domain_scores.iter().map(|(_, score)| score).sum::<f64>()
                / domain_scores.len() as f64;
        }
    }
}

impl Default for AudioProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainMetrics for AudioProcessingMetrics {
    type Result = DomainEvaluationResult;

    fn domain_name(&self) -> &'static str {
        "Audio Processing"
    }

    fn available_metrics(&self) -> Vec<&'static str> {
        vec![
            "word_error_rate",
            "character_error_rate",
            "phone_error_rate",
            "bleu_score",
            "classification_accuracy",
            "classification_f1_score",
            "beat_f_measure",
            "onset_f_measure",
            "chord_recognition_accuracy",
            "key_detection_accuracy",
            "tempo_accuracy",
            "snr_db",
            "pesq_score",
            "stoi_score",
            "speaker_identification_accuracy",
            "speaker_verification_eer",
            "similarity_cosine",
            "similarity_euclidean",
        ]
    }

    fn metric_descriptions(&self) -> HashMap<&'static str, &'static str> {
        let mut descriptions = HashMap::new();
        descriptions.insert(
            "word_error_rate",
            "Word Error Rate for speech recognition evaluation",
        );
        descriptions.insert(
            "character_error_rate",
            "Character Error Rate for detailed speech recognition analysis",
        );
        descriptions.insert(
            "phone_error_rate",
            "Phone Error Rate for phonetic-level speech recognition evaluation",
        );
        descriptions.insert(
            "bleu_score",
            "BLEU score for speech translation quality assessment",
        );
        descriptions.insert(
            "classification_accuracy",
            "Accuracy for audio classification tasks",
        );
        descriptions.insert(
            "classification_f1_score",
            "F1 score for audio classification tasks",
        );
        descriptions.insert(
            "beat_f_measure",
            "F-measure for beat tracking accuracy in music",
        );
        descriptions.insert(
            "onset_f_measure",
            "F-measure for onset detection accuracy in music",
        );
        descriptions.insert(
            "chord_recognition_accuracy",
            "Accuracy for chord recognition in music",
        );
        descriptions.insert(
            "key_detection_accuracy",
            "Accuracy for key detection in music",
        );
        descriptions.insert("tempo_accuracy", "Accuracy for tempo estimation in music");
        descriptions.insert(
            "snr_db",
            "Signal-to-Noise Ratio in decibels for audio quality",
        );
        descriptions.insert("pesq_score", "PESQ score for speech quality assessment");
        descriptions.insert(
            "stoi_score",
            "STOI score for speech intelligibility assessment",
        );
        descriptions.insert(
            "speaker_identification_accuracy",
            "Accuracy for speaker identification",
        );
        descriptions.insert(
            "speaker_verification_eer",
            "Equal Error Rate for speaker verification",
        );
        descriptions.insert(
            "similarity_cosine",
            "Cosine similarity for audio similarity measurement",
        );
        descriptions.insert(
            "similarity_euclidean",
            "Euclidean distance for audio similarity measurement",
        );
        descriptions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_audio_processing_metrics_creation() {
        let _metrics = AudioProcessingMetrics::new();
        // Basic test to ensure creation works
    }

    #[test]
    fn test_speech_recognition_evaluation() {
        let mut metrics = AudioProcessingMetrics::new();
        let reference = vec!["hello world".to_string(), "how are you".to_string()];
        let hypothesis = vec!["hello word".to_string(), "how are you".to_string()];

        let results = metrics
            .evaluate_speech_recognition(&reference, &hypothesis, None, None, None)
            .unwrap();

        assert!(results.wer >= 0.0 && results.wer <= 1.0);
        assert!(results.cer >= 0.0 && results.cer <= 1.0);
    }

    #[test]
    fn test_audio_quality_evaluation() {
        let mut metrics = AudioProcessingMetrics::new();
        // Generate longer signals for PESQ computation (minimum 8000 samples required)
        let reference: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.01).sin()).collect();
        let degraded: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.01).sin() * 0.9).collect();

        let reference = Array1::from_vec(reference);
        let degraded = Array1::from_vec(degraded);

        let results = metrics
            .evaluate_audio_quality(reference.view(), degraded.view(), 16000.0)
            .unwrap();

        assert!(results.snr.is_finite());
        assert!(results.sdr.is_finite());
    }

    #[test]
    fn test_comprehensive_report_creation() {
        let metrics = AudioProcessingMetrics::new();
        let results = AudioEvaluationResults {
            speech_recognition: None,
            audio_classification: None,
            music_information: None,
            quality_assessment: None,
            event_detection: None,
            speaker_recognition: None,
            similarity: None,
        };

        let report = metrics.create_comprehensive_report(&results);
        assert!(!report.summary.best_domain.is_empty());
        assert!(!report.summary.worst_domain.is_empty());
    }
}
