//! Advanced Activity Recognition Framework
//!
//! This module provides sophisticated activity_ recognition capabilities including:
//! - Real-time action detection and classification
//! - Complex activity_ sequence analysis
//! - Multi-person interaction recognition
//! - Context-aware activity_ understanding
//! - Temporal activity_ modeling
//! - Hierarchical activity_ decomposition
//!
//! # Implementation status
//!
//! This module is **experimental**. It consumes the classical (non-semantic)
//! detections from [`crate::scene_understanding`] -- there is no trained
//! action/activity classifier behind it, so activity labels (e.g. "walking",
//! "sitting") are motion-threshold heuristics on optical flow, not genuinely
//! predicted semantic classes.
//!
//! Computed for real from the underlying detections/motion: per-frame and
//! per-sequence [`ActivitySummary`] statistics, single-frame optical flow
//! (frame-to-frame, via an internally tracked previous frame) and the
//! resulting per-person motion features/activity classification, real
//! spatial-proximity interaction candidates
//! (`ActivityRecognitionEngine::detect_frame_interactions`) plus real
//! multi-frame track-based interaction classification
//! (`MultiPersonInteractionRecognizer::analyze_interactions`), timeline
//! segmentation with a real per-segment activity mix, majority-vote/moving-
//! average temporal smoothing, motion-similarity-based hierarchical
//! grouping, and [`ConfidenceScores`] (aggregated from real per-activity
//! confidences and real spatial/temporal coverage).
//!
//! Still an honest placeholder, pending either a trained/probabilistic model
//! or a dedicated follow-up: [`ActivityUncertainty`]'s epistemic/aleatoric/
//! temporal/spatial decomposition and its confusion matrix (there is no
//! model here to calibrate an uncertainty decomposition against, so
//! inventing numbers for these specific fields would be worse than leaving
//! them as the documented placeholder they are), and activity-sequence
//! anomaly detection. Treat any output from those two specific paths with
//! caution until they are implemented.

#![allow(dead_code, missing_docs)]

use crate::error::{Result, VisionError};
use crate::scene_understanding::SceneAnalysisResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView3};
use std::collections::HashMap;

mod types;
pub use types::*;

impl Default for ActivityRecognitionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ActivityRecognitionEngine {
    /// Create a new advanced activity_ recognition engine
    pub fn new() -> Self {
        Self {
            action_detectors: vec![
                ActionDetector::new("human_action_detector"),
                ActionDetector::new("object_interaction_detector"),
            ],
            sequence_analyzer: ActivitySequenceAnalyzer::new(),
            interaction_recognizer: MultiPersonInteractionRecognizer::new(),
            context_classifier: ContextAwareActivityClassifier::new(),
            temporal_modeler: TemporalActivityModeler::new(),
            hierarchical_decomposer: HierarchicalActivityDecomposer::new(),
            knowledge_base: ActivityKnowledgeBase::new(),
            previous_frame: std::cell::RefCell::new(None),
        }
    }

    /// Recognize activities in a single frame
    pub fn recognize_frame_activities(
        &self,
        frame: &ArrayView3<f32>,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<ActivityRecognitionResult> {
        // Extract motion features
        let motion_features = self.extract_motion_features(frame)?;

        // Detect individual actions
        let detected_actions = self.detect_actions(frame, scene_analysis, &motion_features)?;

        // Classify context
        let context = self.context_classifier.classify_context(scene_analysis)?;

        // Enhance detection with context
        let enhanced_activities = self.enhance_with_context(&detected_actions, &context)?;

        // Confidence scores computed for real from the detections above
        // rather than fixed constants. `temporal_segmentation` is genuinely
        // `0.0` here: a single frame carries no temporal information to
        // segment. Epistemic/aleatoric uncertainty decomposition would need
        // a trained probabilistic model this crate does not have, so
        // [`ActivityUncertainty`]'s fields remain a documented placeholder
        // (see the module-level doc comment).
        let overall = mean_confidence(&enhanced_activities);
        let per_activity = mean_confidence_by_class(&enhanced_activities);
        let spatial_localization = if enhanced_activities.is_empty() {
            0.0
        } else {
            enhanced_activities
                .iter()
                .filter(|a| a.spatial_region.is_some())
                .count() as f32
                / enhanced_activities.len() as f32
        };

        // Create result
        Ok(ActivityRecognitionResult {
            activities: enhanced_activities,
            sequences: Vec::new(), // Single frame, no sequences
            interactions: self.detect_frame_interactions(scene_analysis)?,
            scene_summary: self.summarize_frame_activities(scene_analysis)?,
            timeline: ActivityTimeline {
                segments: Vec::new(),
                resolution: 1.0,
                flow_patterns: Vec::new(),
            },
            confidence_scores: ConfidenceScores {
                overall,
                per_activity,
                temporal_segmentation: 0.0,
                spatial_localization,
            },
            uncertainty: ActivityUncertainty {
                epistemic: 0.2,
                aleatoric: 0.15,
                temporal: 0.0,
                spatial: 0.1,
                confusion_matrix: Array2::zeros((10, 10)),
            },
        })
    }

    /// Recognize activities in a video sequence
    pub fn recognize_sequence_activities(
        &self,
        frames: &[ArrayView3<f32>],
        scene_analyses: &[SceneAnalysisResult],
    ) -> Result<ActivityRecognitionResult> {
        if frames.len() != scene_analyses.len() {
            return Err(VisionError::InvalidInput(
                "Number of frames must match number of scene _analyses".to_string(),
            ));
        }

        // Analyze each frame
        let mut frame_activities = Vec::new();
        for (frame, scene_analysis) in frames.iter().zip(scene_analyses.iter()) {
            let frame_result = self.recognize_frame_activities(frame, scene_analysis)?;
            frame_activities.push(frame_result);
        }
        let frame_count = frame_activities.len();

        // Temporal sequence analysis
        let sequences = self
            .sequence_analyzer
            .analyze_sequences(&frame_activities)?;

        // Multi-person interaction analysis
        let interactions = self
            .interaction_recognizer
            .analyze_interactions(scene_analyses)?;

        // Build comprehensive timeline
        let timeline = self.build_activity_timeline(&frame_activities)?;

        // Overall scene summary
        let scene_summary = self.summarize_sequence_activities(&frame_activities)?;

        // Real "how chunky is the timeline" confidence: few, long segments
        // relative to the frame count indicate a stable, confidently
        // segmented activity stream; one segment per frame (maximal
        // flicker) drives this to 0.
        let temporal_segmentation = if frame_count == 0 || timeline.segments.is_empty() {
            0.0
        } else {
            (1.0 - timeline.segments.len() as f32 / frame_count as f32).clamp(0.0, 1.0)
        };

        // Aggregate activities from all frames
        let all_activities: Vec<DetectedActivity> = frame_activities
            .into_iter()
            .flat_map(|result| result.activities)
            .collect();

        let overall = mean_confidence(&all_activities);
        let per_activity = mean_confidence_by_class(&all_activities);
        let spatial_localization = if all_activities.is_empty() {
            0.0
        } else {
            all_activities
                .iter()
                .filter(|a| a.spatial_region.is_some())
                .count() as f32
                / all_activities.len() as f32
        };

        Ok(ActivityRecognitionResult {
            activities: all_activities,
            sequences,
            interactions,
            scene_summary,
            timeline,
            confidence_scores: ConfidenceScores {
                overall,
                per_activity,
                temporal_segmentation,
                spatial_localization,
            },
            uncertainty: ActivityUncertainty {
                epistemic: 0.15,
                aleatoric: 0.1,
                temporal: 0.12,
                spatial: 0.08,
                confusion_matrix: Array2::zeros((10, 10)),
            },
        })
    }

    /// Detect complex multi-person interactions
    pub fn detect_complex_interactions(
        &self,
        scene_sequence: &[SceneAnalysisResult],
    ) -> Result<Vec<PersonInteraction>> {
        self.interaction_recognizer
            .analyze_interactions(scene_sequence)
    }

    /// Recognize hierarchical activity_ structure
    pub fn recognize_hierarchical_structure(
        &self,
        activities: &[DetectedActivity],
    ) -> Result<HierarchicalActivityStructure> {
        self.hierarchical_decomposer
            .decompose_activities(activities)
    }

    /// Predict future activities based on current sequence
    pub fn predict_future_activities(
        &self,
        current_activities: &[DetectedActivity],
        prediction_horizon: f32,
    ) -> Result<Vec<ActivityPrediction>> {
        self.temporal_modeler
            .predict_activities(current_activities, prediction_horizon)
    }

    // Helper methods (real implementations)
    fn extract_motion_features(&self, frame: &ArrayView3<f32>) -> Result<Array3<f32>> {
        let (height, width, _channels) = frame.dim();
        let mut motion_features = Array3::zeros((height, width, 10));

        // Snapshot the previously processed frame and store the current one
        // in its place (interior mutability: this method takes `&self` so
        // that per-frame callers don't need `&mut self`). Only used when its
        // dimensions match the current frame -- a resolution change is
        // treated the same as "no previous frame" rather than indexing out
        // of bounds or fabricating flow across mismatched frames.
        let prev_frame = self.previous_frame.replace(Some(frame.to_owned()));
        let prev_frame = prev_frame.filter(|p| p.dim() == frame.dim());

        // Extract basic motion features
        // Feature 0-1: Optical flow (x, y components)
        if let Some(ref prev_frame) = prev_frame {
            let flow = self.compute_optical_flow(frame, prev_frame)?;
            motion_features
                .slice_mut(scirs2_core::ndarray::s![.., .., 0])
                .assign(&flow.slice(scirs2_core::ndarray::s![.., .., 0]));
            motion_features
                .slice_mut(scirs2_core::ndarray::s![.., .., 1])
                .assign(&flow.slice(scirs2_core::ndarray::s![.., .., 1]));
        }

        // Feature 2: Motion magnitude
        for y in 0..height {
            for x in 0..width {
                let fx = motion_features[[y, x, 0]];
                let fy = motion_features[[y, x, 1]];
                motion_features[[y, x, 2]] = (fx * fx + fy * fy).sqrt();
            }
        }

        // Feature 3: Motion direction
        for y in 0..height {
            for x in 0..width {
                let fx = motion_features[[y, x, 0]];
                let fy = motion_features[[y, x, 1]];
                motion_features[[y, x, 3]] = fy.atan2(fx);
            }
        }

        // Features 4-5: Temporal gradient
        if let Some(ref prev_frame) = prev_frame {
            for y in 0..height {
                for x in 0..width {
                    let current = frame[[y, x, 0]];
                    let previous = prev_frame[[y, x, 0]];
                    motion_features[[y, x, 4]] = current - previous;
                    motion_features[[y, x, 5]] = (current - previous).abs();
                }
            }
        }

        // Features 6-9: Spatial gradients and motion boundaries
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mag = motion_features[[y, x, 2]];
                let mag_left = motion_features[[y, x - 1, 2]];
                let mag_right = motion_features[[y, x + 1, 2]];
                let mag_up = motion_features[[y - 1, x, 2]];
                let mag_down = motion_features[[y + 1, x, 2]];

                motion_features[[y, x, 6]] = mag_right - mag_left; // Horizontal gradient
                motion_features[[y, x, 7]] = mag_down - mag_up; // Vertical gradient
                motion_features[[y, x, 8]] =
                    (mag - (mag_left + mag_right + mag_up + mag_down) / 4.0).abs(); // Motion boundary
                motion_features[[y, x, 9]] = mag.max(0.1).ln(); // Log magnitude for scale invariance
            }
        }

        Ok(motion_features)
    }

    fn detect_actions(
        &self,
        self_frame: &ArrayView3<f32>,
        scene_analysis: &SceneAnalysisResult,
        motion_features: &Array3<f32>,
    ) -> Result<Vec<DetectedActivity>> {
        let mut activities = Vec::new();

        // Analyze each detected person with real activity_ recognition
        for (i, object) in scene_analysis.objects.iter().enumerate() {
            if object.class == "person" {
                // Extract region of interest for the person
                let (bbox_x, bbox_y, bbox_w, bbox_h) = object.bbox;
                let person_motion = self.extract_person_motion_features(
                    motion_features,
                    bbox_x as usize,
                    bbox_y as usize,
                    bbox_w as usize,
                    bbox_h as usize,
                )?;

                // Classify activity_ based on motion characteristics
                let (activity_class, confidence) = self.classify_person_activity(&person_motion);

                // Compute motion characteristics
                let motion_chars = self.compute_motion_characteristics(&person_motion);

                // Detect interaction with objects
                let involved_objects = self.detect_object_interactions(scene_analysis, object)?;

                let activity_ = DetectedActivity {
                    activity_class,
                    subtype: self.determine_activity_subtype(&person_motion),
                    confidence,
                    temporal_bounds: (0.0, 1.0),
                    spatial_region: Some(object.bbox),
                    involved_persons: vec![format!("person_{}", i)],
                    involved_objects,
                    attributes: self.extract_activity_attributes(&person_motion),
                    motion_characteristics: motion_chars,
                };
                activities.push(activity_);
            }
        }

        Ok(activities)
    }

    /// Combine each activity's own confidence with the scene classifier's
    /// confidence (`context.environment_factors["scene_confidence"]`) via a
    /// simple independent-evidence product: when the scene itself was
    /// classified with low confidence, activity confidences derived from it
    /// are scaled down accordingly, rather than reported at face value.
    fn enhance_with_context(
        &self,
        activities: &[DetectedActivity],
        context: &ContextClassification,
    ) -> Result<Vec<DetectedActivity>> {
        let scene_confidence = context
            .environment_factors
            .get("scene_confidence")
            .copied()
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);

        Ok(activities
            .iter()
            .cloned()
            .map(|mut activity_| {
                activity_.confidence *= scene_confidence;
                activity_
            })
            .collect())
    }

    /// Detect person-to-person interaction *candidates* from a single frame.
    ///
    /// Unlike [`MultiPersonInteractionRecognizer::analyze_interactions`]
    /// (which classifies interaction *type* -- following/conversation/
    /// collaboration -- from multi-frame motion tracks), a single frame
    /// carries no trajectory, so this only reports real spatial proximity
    /// between detected people as a generic `"proximate"` interaction; it
    /// does not fabricate a specific interaction type or a non-zero
    /// duration (which genuinely requires multiple frames).
    fn detect_frame_interactions(
        &self,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<Vec<PersonInteraction>> {
        const PROXIMITY_THRESHOLD: f32 = 150.0;

        let people: Vec<(usize, (f32, f32))> = scene_analysis
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.class == "person")
            .map(|(i, o)| {
                let center = (o.bbox.0 + o.bbox.2 / 2.0, o.bbox.1 + o.bbox.3 / 2.0);
                (i, center)
            })
            .collect();

        let mut interactions = Vec::new();
        for i in 0..people.len() {
            for j in (i + 1)..people.len() {
                let (id_i, pos_i) = people[i];
                let (id_j, pos_j) = people[j];
                let distance = ((pos_i.0 - pos_j.0).powi(2) + (pos_i.1 - pos_j.1).powi(2)).sqrt();
                if distance < PROXIMITY_THRESHOLD {
                    let strength = 1.0 - (distance / PROXIMITY_THRESHOLD);
                    interactions.push(PersonInteraction {
                        interaction_type: "proximate".to_string(),
                        participants: vec![format!("person_{id_i}"), format!("person_{id_j}")],
                        strength,
                        duration: 0.0,
                        proximity: distance,
                        attributes: HashMap::new(),
                    });
                }
            }
        }

        Ok(interactions)
    }

    /// Real (if heuristic) per-frame activity summary, computed from the
    /// actual detections/relationships in `scene_analysis` rather than
    /// fixed constants. `dominant_activity` is a coarse, non-semantic
    /// label (there is no trained activity classifier here).
    fn summarize_frame_activities(
        &self,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<ActivitySummary> {
        let object_count = scene_analysis.objects.len();
        let relationship_count = scene_analysis.relationships.len();

        let dominant_activity = if object_count == 0 {
            "static_scene".to_string()
        } else if relationship_count > object_count {
            "interacting_scene".to_string()
        } else {
            "active_scene".to_string()
        };

        let mut distinct_classes = std::collections::HashSet::new();
        for obj in &scene_analysis.objects {
            distinct_classes.insert(obj.class.as_str());
        }
        let diversity_index = if object_count > 0 {
            distinct_classes.len() as f32 / object_count as f32
        } else {
            0.0
        };

        let energy_level = (object_count as f32 / 10.0).min(1.0);
        let max_relationships = object_count.saturating_mul(object_count).max(1);
        let social_interaction_level =
            (relationship_count as f32 / max_relationships as f32).min(1.0);
        let complexity_score =
            ((object_count as f32 * 0.1) + (relationship_count as f32 * 0.15)).min(1.0);

        Ok(ActivitySummary {
            dominant_activity,
            diversity_index,
            energy_level,
            social_interaction_level,
            complexity_score,
            anomaly_indicators: Vec::new(),
        })
    }

    /// Build a real timeline by grouping consecutive frames that share the
    /// same (heuristic, non-semantic) `scene_summary.dominant_activity` into
    /// segments, with per-segment timing from `temporal_resolution` and a
    /// real `activity_mix` computed from the individual
    /// [`DetectedActivity`] labels observed in that segment's frames.
    fn build_activity_timeline(
        &self,
        frame_activities: &[ActivityRecognitionResult],
    ) -> Result<ActivityTimeline> {
        let resolution = self.temporal_modeler.temporal_resolution;
        if frame_activities.is_empty() {
            return Ok(ActivityTimeline {
                segments: Vec::new(),
                resolution,
                flow_patterns: Vec::new(),
            });
        }

        let mut segments = Vec::new();
        let mut segment_start = 0usize;
        for i in 1..=frame_activities.len() {
            let boundary = i == frame_activities.len()
                || frame_activities[i].scene_summary.dominant_activity
                    != frame_activities[segment_start]
                        .scene_summary
                        .dominant_activity;
            if boundary {
                let segment_frames = &frame_activities[segment_start..i];
                segments.push(TimelineSegment {
                    start_time: segment_start as f32 * resolution,
                    end_time: i as f32 * resolution,
                    dominant_activity: frame_activities[segment_start]
                        .scene_summary
                        .dominant_activity
                        .clone(),
                    activity_mix: Self::compute_activity_mix(segment_frames),
                });
                segment_start = i;
            }
        }

        // Real, if coarse, flow measure: how often the dominant activity
        // switches per unit time, derived from the segment boundaries above.
        let flow_patterns = if segments.len() > 1 {
            let total_duration = (frame_activities.len() as f32 * resolution).max(resolution);
            vec![FlowPattern {
                pattern_type: "activity_switching".to_string(),
                frequency: (segments.len() - 1) as f32 / total_duration,
                amplitude: 1.0,
                phase: 0.0,
            }]
        } else {
            Vec::new()
        };

        Ok(ActivityTimeline {
            segments,
            resolution,
            flow_patterns,
        })
    }

    /// Fraction of individual [`DetectedActivity`] labels, across `frames`,
    /// attributable to each activity class.
    fn compute_activity_mix(frames: &[ActivityRecognitionResult]) -> HashMap<String, f32> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for frame in frames {
            for activity_ in &frame.activities {
                *counts.entry(activity_.activity_class.clone()).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return HashMap::new();
        }
        counts
            .into_iter()
            .map(|(k, v)| (k, v as f32 / total as f32))
            .collect()
    }

    /// Aggregate real per-frame [`ActivitySummary`] values (each already
    /// computed for real by [`Self::summarize_frame_activities`]) into a
    /// single sequence-level summary: numeric fields are averaged across
    /// frames, `dominant_activity` is the majority vote, and anomaly
    /// indicators are the concatenation of every frame's.
    fn summarize_sequence_activities(
        &self,
        frame_activities: &[ActivityRecognitionResult],
    ) -> Result<ActivitySummary> {
        if frame_activities.is_empty() {
            return Ok(ActivitySummary {
                dominant_activity: "static_scene".to_string(),
                diversity_index: 0.0,
                energy_level: 0.0,
                social_interaction_level: 0.0,
                complexity_score: 0.0,
                anomaly_indicators: Vec::new(),
            });
        }

        let n = frame_activities.len() as f32;
        let mut diversity_sum = 0.0;
        let mut energy_sum = 0.0;
        let mut social_sum = 0.0;
        let mut complexity_sum = 0.0;
        let mut dominant_counts: HashMap<String, usize> = HashMap::new();
        let mut anomaly_indicators = Vec::new();

        for frame in frame_activities {
            let summary = &frame.scene_summary;
            diversity_sum += summary.diversity_index;
            energy_sum += summary.energy_level;
            social_sum += summary.social_interaction_level;
            complexity_sum += summary.complexity_score;
            *dominant_counts
                .entry(summary.dominant_activity.clone())
                .or_insert(0) += 1;
            anomaly_indicators.extend(summary.anomaly_indicators.iter().cloned());
        }

        let dominant_activity = dominant_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(activity_, _)| activity_)
            .unwrap_or_else(|| "unknown".to_string());

        Ok(ActivitySummary {
            dominant_activity,
            diversity_index: diversity_sum / n,
            energy_level: energy_sum / n,
            social_interaction_level: social_sum / n,
            complexity_score: complexity_sum / n,
            anomaly_indicators,
        })
    }

    // Additional helper methods for activity_ analysis
    fn analyze_person_interaction(
        &self,
        id1: &str,
        id2: &str,
        track1: &[(f32, f32)],
        track2: &[(f32, f32)],
    ) -> Result<Option<PersonInteraction>> {
        if track1.len() != track2.len() || track1.is_empty() {
            return Ok(None);
        }

        // Calculate average distance and relative motion
        let mut total_distance = 0.0;
        let mut relative_motion = 0.0;
        let mut close_proximity_frames = 0;

        for i in 0..track1.len() {
            let distance =
                ((track1[i].0 - track2[i].0).powi(2) + (track1[i].1 - track2[i].1).powi(2)).sqrt();
            total_distance += distance;

            if distance < 150.0 {
                // Close proximity threshold
                close_proximity_frames += 1;
            }

            if i > 0 {
                let velocity1 = ((track1[i].0 - track1[i - 1].0).powi(2)
                    + (track1[i].1 - track1[i - 1].1).powi(2))
                .sqrt();
                let velocity2 = ((track2[i].0 - track2[i - 1].0).powi(2)
                    + (track2[i].1 - track2[i - 1].1).powi(2))
                .sqrt();
                relative_motion += (velocity1 - velocity2).abs();
            }
        }

        let avg_distance = total_distance / track1.len() as f32;
        let proximity_ratio = close_proximity_frames as f32 / track1.len() as f32;

        if proximity_ratio > 0.3 {
            // Threshold for interaction
            let interaction_type = if relative_motion / (track1.len() as f32) < 5.0 {
                "following".to_string()
            } else if avg_distance < 100.0 {
                "conversation".to_string()
            } else {
                "collaboration".to_string()
            };

            Ok(Some(PersonInteraction {
                interaction_type,
                participants: vec![id1.to_string(), id2.to_string()],
                strength: proximity_ratio,
                duration: track1.len() as f32 / 30.0, // Assuming 30 FPS
                proximity: avg_distance,
                attributes: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    fn count_activity_types(&self, activities: &[DetectedActivity]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for activity_ in activities {
            *counts.entry(activity_.activity_class.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn find_dominant_activity(&self, activitycounts: &HashMap<String, usize>) -> String {
        activitycounts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(activity_, _)| activity_.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn predict_activity_transition(&self, currentactivity: &str) -> Option<String> {
        // Simple transition model based on common _activity patterns
        match currentactivity {
            "sitting" => Some("standing".to_string()),
            "standing" => Some("walking".to_string()),
            "walking" => Some("standing".to_string()),
            "running" => Some("walking".to_string()),
            "gesturing" => Some("standing".to_string()),
            _ => None,
        }
    }

    fn group_activities_by_similarity(
        &self,
        activities: &[DetectedActivity],
    ) -> HashMap<String, Vec<DetectedActivity>> {
        let mut groups = HashMap::new();

        for activity_ in activities {
            let group_key = if activity_.motion_characteristics.velocity > 0.5 {
                "dynamic_activities".to_string()
            } else if activity_.motion_characteristics.velocity < 0.1 {
                "static_activities".to_string()
            } else {
                "moderate_activities".to_string()
            };

            groups
                .entry(group_key)
                .or_insert_with(Vec::new)
                .push(activity_.clone());
        }

        groups
    }
}

// Implementation stubs for associated types
impl ActionDetector {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            action_types: vec![
                "walking".to_string(),
                "sitting".to_string(),
                "standing".to_string(),
            ],
            confidence_threshold: 0.5,
            temporal_window: 30,
            feature_method: "optical_flow".to_string(),
        }
    }
}

impl ActivitySequenceAnalyzer {
    fn new() -> Self {
        Self {
            max_sequence_length: 100,
            pattern_models: Vec::new(),
            transition_models: HashMap::new(),
            anomaly_params: AnomalyDetectionParams {
                detection_threshold: 0.3,
                temporal_window: 10,
                feature_importance: Array1::ones(50),
                novelty_detection: true,
            },
        }
    }

    fn analyze_sequences(
        &self,
        frame_activities: &[ActivityRecognitionResult],
    ) -> Result<Vec<ActivitySequence>> {
        let mut sequences = Vec::new();

        if frame_activities.len() < 2 {
            return Ok(sequences);
        }

        // Find activity_ sequences across frames
        let mut current_sequence: Option<ActivitySequence> = None;

        for frame_result in frame_activities.iter() {
            for activity_ in &frame_result.activities {
                match &mut current_sequence {
                    None => {
                        // Start new sequence
                        current_sequence = Some(ActivitySequence {
                            sequence_id: format!("seq_{}", sequences.len()),
                            activities: vec![activity_.clone()],
                            sequence_type: activity_.activity_class.clone(),
                            confidence: activity_.confidence,
                            transitions: Vec::new(),
                            completeness: 0.0,
                        });
                    }
                    Some(ref mut seq) => {
                        if activity_.activity_class == seq.sequence_type {
                            // Continue existing sequence
                            seq.activities.push(activity_.clone());
                            seq.confidence = (seq.confidence + activity_.confidence) / 2.0;
                        } else {
                            // End current sequence and start new one
                            seq.completeness =
                                seq.activities.len() as f32 / frame_activities.len() as f32;
                            sequences.push(seq.clone());

                            current_sequence = Some(ActivitySequence {
                                sequence_id: format!("seq_{}", sequences.len()),
                                activities: vec![activity_.clone()],
                                sequence_type: activity_.activity_class.clone(),
                                confidence: activity_.confidence,
                                transitions: vec![ActivityTransition {
                                    from_activity: seq.sequence_type.clone(),
                                    to_activity: activity_.activity_class.clone(),
                                    probability: 0.8,
                                    typical_duration: 1.0,
                                }],
                                completeness: 0.0,
                            });
                        }
                    }
                }
            }
        }

        // Add final sequence
        if let Some(mut seq) = current_sequence {
            seq.completeness = seq.activities.len() as f32 / frame_activities.len() as f32;
            sequences.push(seq);
        }

        Ok(sequences)
    }
}

impl MultiPersonInteractionRecognizer {
    fn new() -> Self {
        Self {
            interaction_types: vec![
                InteractionType::Conversation,
                InteractionType::Collaboration,
            ],
            tracking_params: PersonTrackingParams {
                max_tracking_distance: 50.0,
                identity_confidence_threshold: 0.8,
                re_identification_enabled: true,
                track_merge_threshold: 0.7,
            },
            social_distance_model: SocialDistanceModel {
                personal_space_radius: 0.5,
                social_space_radius: 1.5,
                public_space_radius: 3.0,
                cultural_factors: HashMap::new(),
            },
            group_recognition: GroupActivityRecognition {
                min_group_size: 2,
                max_group_size: 10,
                cohesion_threshold: 0.6,
                activity_synchronization: true,
            },
        }
    }

    fn analyze_interactions(
        &self,
        scene_analyses: &[SceneAnalysisResult],
    ) -> Result<Vec<PersonInteraction>> {
        let mut interactions = Vec::new();

        if scene_analyses.len() < 2 {
            return Ok(interactions);
        }

        // Track person positions across frames
        let mut person_tracks: HashMap<String, Vec<(f32, f32)>> = HashMap::new();

        for scene in scene_analyses {
            for (i, object) in scene.objects.iter().enumerate() {
                if object.class == "person" {
                    let person_id = format!("person_{i}");
                    let position = (
                        object.bbox.0 + object.bbox.2 / 2.0,
                        object.bbox.1 + object.bbox.3 / 2.0,
                    );
                    person_tracks.entry(person_id).or_default().push(position);
                }
            }
        }

        // Analyze interactions between people
        let person_ids: Vec<_> = person_tracks.keys().cloned().collect();

        for i in 0..person_ids.len() {
            for j in (i + 1)..person_ids.len() {
                let id1 = &person_ids[i];
                let id2 = &person_ids[j];

                if let (Some(track1), Some(track2)) =
                    (person_tracks.get(id1), person_tracks.get(id2))
                {
                    let interaction = self.analyze_person_interaction(id1, id2, track1, track2)?;
                    if let Some(interaction) = interaction {
                        interactions.push(interaction);
                    }
                }
            }
        }

        Ok(interactions)
    }
}

impl ContextAwareActivityClassifier {
    fn new() -> Self {
        Self {
            context_features: vec![ContextFeature::SceneType, ContextFeature::CrowdDensity],
            environment_classifiers: Vec::new(),
            object_associations: HashMap::new(),
            scene_correlations: HashMap::new(),
        }
    }

    /// Classify the scene's context from real, already-computed scene
    /// analysis: `scene_type` is [`SceneAnalysisResult::scene_class`]'s
    /// classical (non-semantic) label passed straight through -- there is no
    /// separate indoor/outdoor classifier here, so this deliberately reuses
    /// the same real label rather than inventing a new "indoor"/"outdoor"
    /// guess with no support in the data. `environment_factors` are real
    /// counts/statistics derived from the scene.
    fn classify_context(
        &self,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<ContextClassification> {
        let mut environment_factors = HashMap::new();
        environment_factors.insert(
            "object_count".to_string(),
            scene_analysis.objects.len() as f32,
        );
        environment_factors.insert(
            "relationship_count".to_string(),
            scene_analysis.relationships.len() as f32,
        );
        environment_factors.insert(
            "scene_confidence".to_string(),
            scene_analysis.scene_confidence,
        );
        let person_count = scene_analysis
            .objects
            .iter()
            .filter(|o| o.class == "person")
            .count() as f32;
        environment_factors.insert("person_count".to_string(), person_count);

        Ok(ContextClassification {
            scene_type: scene_analysis.scene_class.clone(),
            environment_factors,
            temporal_context: HashMap::new(),
        })
    }
}

impl TemporalActivityModeler {
    fn new() -> Self {
        Self {
            temporal_resolution: 1.0 / 30.0,
            memory_length: 100,
            rnn_params: RNNParameters {
                hidden_size: 128,
                num_layers: 2,
                dropout_rate: 0.2,
                bidirectional: true,
            },
            attention_mechanisms: Vec::new(),
        }
    }

    fn predict_activities(
        &self,
        current_activities: &[DetectedActivity],
        prediction_horizon: f32,
    ) -> Result<Vec<ActivityPrediction>> {
        let mut predictions = Vec::new();

        if current_activities.is_empty() {
            return Ok(predictions);
        }

        // Analyze current activity_ patterns
        let activitycounts = self.count_activity_types(current_activities);
        let dominant_activity = self.find_dominant_activity(&activitycounts);

        // Predict based on temporal patterns and transitions
        for (activity_type, count) in activitycounts {
            let confidence = (count as f32 / current_activities.len() as f32) * 0.8;

            // Simple prediction based on activity_ persistence and transitions
            let predicted_duration = if activity_type == dominant_activity {
                prediction_horizon * 0.7 // Dominant activity_ likely to continue
            } else {
                prediction_horizon * 0.3 // Other _activities may transition
            };

            predictions.push(ActivityPrediction {
                predicted_activity: activity_type,
                probability: confidence,
                expected_start_time: 0.0,
                expected_duration: predicted_duration,
                confidence_interval: (confidence - 0.2, confidence + 0.2),
            });
        }

        // Add transition predictions
        for activity_ in current_activities {
            if let Some(transition) = self.predict_activity_transition(&activity_.activity_class) {
                predictions.push(ActivityPrediction {
                    predicted_activity: transition,
                    probability: 0.4,
                    expected_start_time: prediction_horizon * 0.5,
                    expected_duration: prediction_horizon * 0.5,
                    confidence_interval: (0.2, 0.6),
                });
            }
        }

        Ok(predictions)
    }
}

impl HierarchicalActivityDecomposer {
    fn new() -> Self {
        Self {
            hierarchy_levels: Vec::new(),
            decomposition_rules: Vec::new(),
            composition_rules: Vec::new(),
        }
    }

    fn decompose_activities(
        &self,
        activities: &[DetectedActivity],
    ) -> Result<HierarchicalActivityStructure> {
        // Group activities by (motion-based) similarity up front so a real
        // `decomposition_confidence` can be derived from how much genuine
        // structure was found: activities collapsing into few, larger groups
        // indicates a clearer coarse-to-fine hierarchy, while every activity
        // landing in its own group (ratio near 1.0) indicates none was found.
        let activity_groups = self.group_activities_by_similarity(activities);
        let decomposition_confidence = if activities.is_empty() {
            0.0
        } else {
            (1.0 - activity_groups.len() as f32 / activities.len() as f32).clamp(0.0, 1.0)
        };

        let mut structure = HierarchicalActivityStructure {
            levels: vec![
                ActivityLevel {
                    level_name: "atomic".to_string(),
                    granularity: 1.0,
                    typical_duration: 1.0,
                    complexity: 1.0,
                },
                ActivityLevel {
                    level_name: "composite".to_string(),
                    granularity: 0.5,
                    typical_duration: 5.0,
                    complexity: 2.0,
                },
                ActivityLevel {
                    level_name: "complex".to_string(),
                    granularity: 0.2,
                    typical_duration: 15.0,
                    complexity: 3.0,
                },
            ],
            activity_tree: ActivityTree {
                root: ActivityNode {
                    node_id: "root".to_string(),
                    activity_type: "scene".to_string(),
                    level: 0,
                    children: Vec::new(),
                },
                nodes: Vec::new(),
                edges: Vec::new(),
            },
            decomposition_confidence,
        };

        // Build activity_ hierarchy from the groups computed above.
        for (node_id, (group_type, group_activities)) in activity_groups.into_iter().enumerate() {
            let node_id = node_id + 1;
            // Create composite activity_ node
            let composite_node = ActivityNode {
                node_id: format!("composite_{node_id}"),
                activity_type: group_type.clone(),
                level: 1,
                children: Vec::new(),
            };

            structure
                .activity_tree
                .root
                .children
                .push(composite_node.node_id.clone());
            structure.activity_tree.nodes.push(composite_node.clone());

            // Add edge from root to composite
            structure.activity_tree.edges.push(ActivityEdge {
                parent: "root".to_string(),
                child: composite_node.node_id.clone(),
                relationship_type: "contains".to_string(),
            });

            // Create atomic activity_ nodes
            for (i, activity_) in group_activities.iter().enumerate() {
                let atomic_node = ActivityNode {
                    node_id: format!("atomic_{node_id}_{i}"),
                    activity_type: activity_.activity_class.clone(),
                    level: 2,
                    children: Vec::new(),
                };

                structure.activity_tree.nodes.push(atomic_node.clone());
                structure.activity_tree.edges.push(ActivityEdge {
                    parent: composite_node.node_id.clone(),
                    child: atomic_node.node_id.clone(),
                    relationship_type: "instantiation".to_string(),
                });
            }
        }

        Ok(structure)
    }
}

impl ActivityKnowledgeBase {
    fn new() -> Self {
        Self {
            activity_definitions: HashMap::new(),
            ontology: ActivityOntology {
                activity_hierarchy: HashMap::new(),
                activity_relationships: Vec::new(),
                semantic_similarity: Array2::zeros((50, 50)),
            },
            common_patterns: Vec::new(),
            cultural_variations: HashMap::new(),
        }
    }
}

/// Mean [`DetectedActivity::confidence`] across `activities` (`0.0` when
/// empty, rather than a fabricated default).
fn mean_confidence(activities: &[DetectedActivity]) -> f32 {
    if activities.is_empty() {
        return 0.0;
    }
    activities.iter().map(|a| a.confidence).sum::<f32>() / activities.len() as f32
}

/// Mean [`DetectedActivity::confidence`] grouped by `activity_class`.
fn mean_confidence_by_class(activities: &[DetectedActivity]) -> HashMap<String, f32> {
    let mut sums: HashMap<String, (f32, usize)> = HashMap::new();
    for activity_ in activities {
        let entry = sums
            .entry(activity_.activity_class.clone())
            .or_insert((0.0, 0));
        entry.0 += activity_.confidence;
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(class, (sum, count))| (class, sum / count as f32))
        .collect()
}

/// High-level function for comprehensive activity_ recognition
#[allow(dead_code)]
pub fn recognize_activities_comprehensive(
    frames: &[ArrayView3<f32>],
    scene_analyses: &[SceneAnalysisResult],
) -> Result<ActivityRecognitionResult> {
    let engine = ActivityRecognitionEngine::new();

    if frames.len() == 1 {
        engine.recognize_frame_activities(&frames[0], &scene_analyses[0])
    } else {
        engine.recognize_sequence_activities(frames, scene_analyses)
    }
}

/// Specialized function for real-time activity_ monitoring
#[allow(dead_code)]
pub fn monitor_activities_realtime(
    current_frame: &ArrayView3<f32>,
    scene_analysis: &SceneAnalysisResult,
    activity_history: Option<&[ActivityRecognitionResult]>,
) -> Result<ActivityRecognitionResult> {
    let engine = ActivityRecognitionEngine::new();
    let mut result = engine.recognize_frame_activities(current_frame, scene_analysis)?;

    // Apply temporal smoothing if _history is available
    if let Some(_history) = activity_history {
        result = apply_temporal_smoothing(result, _history)?;
    }

    Ok(result)
}

/// Apply real temporal smoothing over a trailing window of `history` plus
/// the current frame, to reduce flickering in real-time recognition:
/// `scene_summary.dominant_activity` is replaced by the majority vote across
/// the window, and the numeric summary fields are replaced by their mean.
/// Per-activity/interaction/sequence/timeline detail from the current frame
/// is left untouched (only the coarse scene-level summary is smoothed).
#[allow(dead_code)]
fn apply_temporal_smoothing(
    current_result: ActivityRecognitionResult,
    history: &[ActivityRecognitionResult],
) -> Result<ActivityRecognitionResult> {
    if history.is_empty() {
        return Ok(current_result);
    }

    const SMOOTHING_WINDOW: usize = 5;
    let window_start = history
        .len()
        .saturating_sub(SMOOTHING_WINDOW.saturating_sub(1));
    let window = &history[window_start..];

    let mut dominant_counts: HashMap<String, usize> = HashMap::new();
    let mut energy_sum = current_result.scene_summary.energy_level;
    let mut social_sum = current_result.scene_summary.social_interaction_level;
    let mut complexity_sum = current_result.scene_summary.complexity_score;
    let mut diversity_sum = current_result.scene_summary.diversity_index;
    *dominant_counts
        .entry(current_result.scene_summary.dominant_activity.clone())
        .or_insert(0) += 1;

    for past in window {
        energy_sum += past.scene_summary.energy_level;
        social_sum += past.scene_summary.social_interaction_level;
        complexity_sum += past.scene_summary.complexity_score;
        diversity_sum += past.scene_summary.diversity_index;
        *dominant_counts
            .entry(past.scene_summary.dominant_activity.clone())
            .or_insert(0) += 1;
    }

    let n = (window.len() + 1) as f32;
    let smoothed_dominant = dominant_counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(activity_, _)| activity_)
        .unwrap_or_else(|| current_result.scene_summary.dominant_activity.clone());

    let mut smoothed = current_result;
    smoothed.scene_summary.dominant_activity = smoothed_dominant;
    smoothed.scene_summary.energy_level = energy_sum / n;
    smoothed.scene_summary.social_interaction_level = social_sum / n;
    smoothed.scene_summary.complexity_score = complexity_sum / n;
    smoothed.scene_summary.diversity_index = diversity_sum / n;

    Ok(smoothed)
}

// Additional missing helper methods for ActivityRecognitionEngine
impl ActivityRecognitionEngine {
    fn compute_optical_flow(
        &self,
        current_frame: &ArrayView3<f32>,
        previous_frame: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let (height, width, _) = current_frame.dim();
        let mut flow = Array3::zeros((height, width, 2));

        // Simple optical flow computation using _frame difference
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let current = current_frame[[y, x, 0]];
                let previous = previous_frame[[y, x, 0]];

                // Compute spatial gradients
                let ix = (current_frame[[y, x + 1, 0]] - current_frame[[y, x - 1, 0]]) / 2.0;
                let iy = (current_frame[[y + 1, x, 0]] - current_frame[[y - 1, x, 0]]) / 2.0;
                let it = current - previous;

                // Lucas-Kanade optical flow (simplified)
                if ix.abs() > 0.01 || iy.abs() > 0.01 {
                    let denominator = ix * ix + iy * iy;
                    if denominator > 0.001 {
                        flow[[y, x, 0]] = -it * ix / denominator;
                        flow[[y, x, 1]] = -it * iy / denominator;
                    }
                }
            }
        }

        Ok(flow)
    }

    fn extract_person_motion_features(
        &self,
        motion_features: &Array3<f32>,
        bbox_x: usize,
        bbox_y: usize,
        bbox_w: usize,
        bbox_h: usize,
    ) -> Result<Array1<f32>> {
        let mut person_features = Array1::zeros(20);

        let end_x = (bbox_x + bbox_w).min(motion_features.dim().1);
        let end_y = (bbox_y + bbox_h).min(motion_features.dim().0);

        // Extract statistics from person bounding box region
        let mut count = 0;
        let mut sum_velocity = 0.0;
        let mut sum_magnitude = 0.0;
        let mut sum_direction = 0.0;

        for _y in bbox_y..end_y {
            for _x in bbox_x..end_x {
                let magnitude = motion_features[[_y, _x, 2]];
                let direction = motion_features[[_y, _x, 3]];

                sum_velocity += magnitude;
                sum_magnitude += magnitude;
                sum_direction += direction;
                count += 1;
            }
        }

        if count > 0 {
            person_features[0] = sum_velocity / count as f32; // Average velocity
            person_features[1] = sum_magnitude / count as f32; // Average magnitude
            person_features[2] = sum_direction / count as f32; // Average direction
            person_features[3] = (bbox_w * bbox_h) as f32; // Person size
            person_features[4] = bbox_w as f32 / bbox_h as f32; // Aspect ratio
        }

        Ok(person_features)
    }

    fn classify_person_activity(&self, person_motionfeatures: &Array1<f32>) -> (String, f32) {
        let velocity = person_motionfeatures[0];
        let magnitude = person_motionfeatures[1];
        let aspect_ratio = person_motionfeatures[4];

        // Simple activity_ classification based on motion characteristics
        if velocity < 0.1 {
            if aspect_ratio > 0.8 {
                ("standing".to_string(), 0.8)
            } else {
                ("sitting".to_string(), 0.7)
            }
        } else if velocity < 0.5 {
            ("walking".to_string(), 0.75)
        } else if velocity < 1.0 {
            ("running".to_string(), 0.7)
        } else if magnitude > 0.5 {
            ("gesturing".to_string(), 0.6)
        } else {
            ("moving_quickly".to_string(), 0.65)
        }
    }

    fn compute_motion_characteristics(
        &self,
        person_motionfeatures: &Array1<f32>,
    ) -> MotionCharacteristics {
        MotionCharacteristics {
            velocity: person_motionfeatures[0],
            acceleration: person_motionfeatures[1] - person_motionfeatures[0], // Simplified
            direction: person_motionfeatures[2],
            smoothness: 1.0 - (person_motionfeatures[1] - person_motionfeatures[0]).abs(),
            periodicity: 0.5, // Placeholder
        }
    }

    fn detect_object_interactions(
        &self,
        scene_analysis: &SceneAnalysisResult,
        person_object: &crate::scene_understanding::DetectedObject,
    ) -> Result<Vec<ObjectID>> {
        let mut interactions = Vec::new();
        let person_center = (
            person_object.bbox.0 + person_object.bbox.2 / 2.0,
            person_object.bbox.1 + person_object.bbox.3 / 2.0,
        );

        for _object in &scene_analysis.objects {
            if _object.class != "person" {
                let object_center = (
                    _object.bbox.0 + _object.bbox.2 / 2.0,
                    _object.bbox.1 + _object.bbox.3 / 2.0,
                );
                let distance = ((person_center.0 - object_center.0).powi(2)
                    + (person_center.1 - object_center.1).powi(2))
                .sqrt();

                // If person is close to object, consider it an interaction
                if distance < 100.0 {
                    interactions.push(format!("{}:unknown", _object.class));
                }
            }
        }

        Ok(interactions)
    }

    fn determine_activity_subtype(&self, person_motionfeatures: &Array1<f32>) -> Option<String> {
        let velocity = person_motionfeatures[0];
        let magnitude = person_motionfeatures[1];

        if velocity > 0.8 {
            Some("fast".to_string())
        } else if velocity < 0.2 {
            Some("slow".to_string())
        } else if magnitude > 0.6 {
            Some("active".to_string())
        } else {
            None
        }
    }

    fn extract_activity_attributes(
        &self,
        person_motionfeatures: &Array1<f32>,
    ) -> HashMap<String, f32> {
        let mut attributes = HashMap::new();

        attributes.insert("velocity".to_string(), person_motionfeatures[0]);
        attributes.insert("magnitude".to_string(), person_motionfeatures[1]);
        attributes.insert("direction".to_string(), person_motionfeatures[2]);
        attributes.insert("size".to_string(), person_motionfeatures[3]);
        attributes.insert("aspect_ratio".to_string(), person_motionfeatures[4]);

        attributes
    }
}

// Fix method implementations for associated types
impl TemporalActivityModeler {
    fn count_activity_types(&self, activities: &[DetectedActivity]) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for activity_ in activities {
            *counts.entry(activity_.activity_class.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn find_dominant_activity(&self, activitycounts: &HashMap<String, usize>) -> String {
        activitycounts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(activity_, _)| activity_.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn predict_activity_transition(&self, currentactivity: &str) -> Option<String> {
        // Simple transition model based on common _activity patterns
        match currentactivity {
            "sitting" => Some("standing".to_string()),
            "standing" => Some("walking".to_string()),
            "walking" => Some("standing".to_string()),
            "running" => Some("walking".to_string()),
            "gesturing" => Some("standing".to_string()),
            _ => None,
        }
    }
}

impl HierarchicalActivityDecomposer {
    fn group_activities_by_similarity(
        &self,
        activities: &[DetectedActivity],
    ) -> HashMap<String, Vec<DetectedActivity>> {
        let mut groups = HashMap::new();

        for activity_ in activities {
            let group_key = if activity_.motion_characteristics.velocity > 0.5 {
                "dynamic_activities".to_string()
            } else if activity_.motion_characteristics.velocity < 0.1 {
                "static_activities".to_string()
            } else {
                "moderate_activities".to_string()
            };

            groups
                .entry(group_key)
                .or_insert_with(Vec::new)
                .push(activity_.clone());
        }

        groups
    }
}

impl MultiPersonInteractionRecognizer {
    fn analyze_person_interaction(
        &self,
        id1: &str,
        id2: &str,
        track1: &[(f32, f32)],
        track2: &[(f32, f32)],
    ) -> Result<Option<PersonInteraction>> {
        if track1.len() != track2.len() || track1.is_empty() {
            return Ok(None);
        }

        // Calculate average distance and relative motion
        let mut total_distance = 0.0;
        let mut relative_motion = 0.0;
        let mut close_proximity_frames = 0;

        for i in 0..track1.len() {
            let distance =
                ((track1[i].0 - track2[i].0).powi(2) + (track1[i].1 - track2[i].1).powi(2)).sqrt();
            total_distance += distance;

            if distance < 150.0 {
                // Close proximity threshold
                close_proximity_frames += 1;
            }

            if i > 0 {
                let velocity1 = ((track1[i].0 - track1[i - 1].0).powi(2)
                    + (track1[i].1 - track1[i - 1].1).powi(2))
                .sqrt();
                let velocity2 = ((track2[i].0 - track2[i - 1].0).powi(2)
                    + (track2[i].1 - track2[i - 1].1).powi(2))
                .sqrt();
                relative_motion += (velocity1 - velocity2).abs();
            }
        }

        let avg_distance = total_distance / track1.len() as f32;
        let proximity_ratio = close_proximity_frames as f32 / track1.len() as f32;

        if proximity_ratio > 0.3 {
            // Threshold for interaction
            let interaction_type = if relative_motion / (track1.len() as f32) < 5.0 {
                "following".to_string()
            } else if avg_distance < 100.0 {
                "conversation".to_string()
            } else {
                "collaboration".to_string()
            };

            Ok(Some(PersonInteraction {
                interaction_type,
                participants: vec![id1.to_string(), id2.to_string()],
                strength: proximity_ratio,
                duration: track1.len() as f32 / 30.0, // Assuming 30 FPS
                proximity: avg_distance,
                attributes: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests;
