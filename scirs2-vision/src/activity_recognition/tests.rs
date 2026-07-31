use super::*;
use crate::scene_understanding::{
    DetectedObject, SceneGraph, SpatialRelation, SpatialRelationType,
};
use scirs2_core::ndarray::Array2;

fn object(bbox: (f32, f32, f32, f32)) -> DetectedObject {
    DetectedObject {
        class: "object".to_string(),
        bbox,
        confidence: 0.9,
        features: Array2::zeros((1, 4)),
        mask: None,
        attributes: HashMap::new(),
    }
}

fn relation(source_id: usize, target_id: usize) -> SpatialRelation {
    SpatialRelation {
        source_id,
        target_id,
        relation_type: SpatialRelationType::NextTo,
        confidence: 0.7,
        parameters: HashMap::new(),
    }
}

fn scene_with(
    objects: Vec<DetectedObject>,
    relationships: Vec<SpatialRelation>,
) -> SceneAnalysisResult {
    SceneAnalysisResult {
        objects,
        relationships,
        scene_class: "test_scene".to_string(),
        scene_confidence: 0.8,
        segmentation_map: Array2::zeros((2, 2)),
        scene_graph: SceneGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            global_properties: HashMap::new(),
        },
        temporal_info: None,
        reasoning_results: Vec::new(),
    }
}

#[test]
fn test_summarize_frame_activities_responds_to_content() {
    // Regression guard: the original implementation returned the exact
    // same hardcoded `ActivitySummary` regardless of `scene_analysis`.
    let engine = ActivityRecognitionEngine::new();

    let empty_scene = scene_with(Vec::new(), Vec::new());
    let empty_summary = engine
        .summarize_frame_activities(&empty_scene)
        .expect("summarize_frame_activities failed");
    assert_eq!(empty_summary.dominant_activity, "static_scene");
    assert_eq!(empty_summary.energy_level, 0.0);

    let busy_objects = vec![
        object((0.0, 0.0, 1.0, 1.0)),
        object((5.0, 5.0, 1.0, 1.0)),
        object((10.0, 10.0, 1.0, 1.0)),
    ];
    let busy_relationships = vec![
        relation(0, 1),
        relation(1, 2),
        relation(0, 2),
        relation(2, 0),
    ];
    let busy_scene = scene_with(busy_objects, busy_relationships);
    let busy_summary = engine
        .summarize_frame_activities(&busy_scene)
        .expect("summarize_frame_activities failed");

    assert_ne!(
        busy_summary.dominant_activity,
        empty_summary.dominant_activity
    );
    assert!(busy_summary.energy_level > empty_summary.energy_level);
    assert_eq!(busy_summary.dominant_activity, "interacting_scene");
    assert!(busy_summary.social_interaction_level > 0.0);
}

fn person(bbox: (f32, f32, f32, f32)) -> DetectedObject {
    DetectedObject {
        class: "person".to_string(),
        bbox,
        confidence: 0.9,
        features: Array2::zeros((1, 4)),
        mask: None,
        attributes: HashMap::new(),
    }
}

fn detected_activity(class: &str, confidence: f32) -> DetectedActivity {
    DetectedActivity {
        activity_class: class.to_string(),
        subtype: None,
        confidence,
        temporal_bounds: (0.0, 1.0),
        spatial_region: Some((0.0, 0.0, 1.0, 1.0)),
        involved_persons: Vec::new(),
        involved_objects: Vec::new(),
        attributes: HashMap::new(),
        motion_characteristics: MotionCharacteristics {
            velocity: 0.0,
            acceleration: 0.0,
            direction: 0.0,
            smoothness: 1.0,
            periodicity: 0.0,
        },
    }
}

fn frame_result(
    dominant_activity: &str,
    energy_level: f32,
    activities: Vec<DetectedActivity>,
) -> ActivityRecognitionResult {
    ActivityRecognitionResult {
        activities,
        sequences: Vec::new(),
        interactions: Vec::new(),
        scene_summary: ActivitySummary {
            dominant_activity: dominant_activity.to_string(),
            diversity_index: 0.0,
            energy_level,
            social_interaction_level: 0.0,
            complexity_score: 0.0,
            anomaly_indicators: Vec::new(),
        },
        timeline: ActivityTimeline {
            segments: Vec::new(),
            resolution: 1.0 / 30.0,
            flow_patterns: Vec::new(),
        },
        confidence_scores: ConfidenceScores {
            overall: 0.0,
            per_activity: HashMap::new(),
            temporal_segmentation: 0.0,
            spatial_localization: 0.0,
        },
        uncertainty: ActivityUncertainty {
            epistemic: 0.0,
            aleatoric: 0.0,
            temporal: 0.0,
            spatial: 0.0,
            confusion_matrix: Array2::zeros((1, 1)),
        },
    }
}

#[test]
fn test_summarize_sequence_activities_aggregates_real_frames_not_hardcoded() {
    // Regression guard: the original implementation returned the exact
    // same hardcoded `ActivitySummary` regardless of the frames.
    let engine = ActivityRecognitionEngine::new();
    let frames = vec![
        frame_result("walking", 0.2, vec![]),
        frame_result("walking", 0.4, vec![]),
        frame_result("sitting", 0.0, vec![]),
    ];
    let summary = engine
        .summarize_sequence_activities(&frames)
        .expect("summarize_sequence_activities failed");

    assert_eq!(summary.dominant_activity, "walking"); // 2 of 3 frames
    assert!(
        (summary.energy_level - 0.2).abs() < 1e-6,
        "got {}",
        summary.energy_level
    );
    assert_ne!(summary.dominant_activity, "general_activity");
}

#[test]
fn test_build_activity_timeline_groups_consecutive_frames_not_empty() {
    let engine = ActivityRecognitionEngine::new();
    let frames = vec![
        frame_result("walking", 0.2, vec![detected_activity("walking", 0.8)]),
        frame_result("walking", 0.3, vec![detected_activity("walking", 0.9)]),
        frame_result("sitting", 0.0, vec![detected_activity("sitting", 0.7)]),
    ];
    let timeline = engine
        .build_activity_timeline(&frames)
        .expect("build_activity_timeline failed");

    assert_eq!(
        timeline.segments.len(),
        2,
        "expected 2 real segments, not an empty placeholder"
    );
    assert_eq!(timeline.segments[0].dominant_activity, "walking");
    assert_eq!(timeline.segments[1].dominant_activity, "sitting");
    let resolution = timeline.resolution;
    assert!((timeline.segments[0].start_time - 0.0).abs() < 1e-6);
    assert!((timeline.segments[0].end_time - 2.0 * resolution).abs() < 1e-6);
    assert!(
        (timeline.segments[0]
            .activity_mix
            .get("walking")
            .copied()
            .unwrap_or(0.0)
            - 1.0)
            .abs()
            < 1e-6
    );
}

#[test]
fn test_detect_frame_interactions_finds_close_people_not_far() {
    let engine = ActivityRecognitionEngine::new();

    let close_scene = scene_with(
        vec![
            person((0.0, 0.0, 10.0, 10.0)),
            person((40.0, 0.0, 10.0, 10.0)),
        ],
        Vec::new(),
    );
    let close_interactions = engine
        .detect_frame_interactions(&close_scene)
        .expect("detect_frame_interactions failed");
    assert_eq!(close_interactions.len(), 1);
    assert_eq!(close_interactions[0].interaction_type, "proximate");

    let far_scene = scene_with(
        vec![
            person((0.0, 0.0, 10.0, 10.0)),
            person((1000.0, 0.0, 10.0, 10.0)),
        ],
        Vec::new(),
    );
    let far_interactions = engine
        .detect_frame_interactions(&far_scene)
        .expect("detect_frame_interactions failed");
    assert!(
        far_interactions.is_empty(),
        "distant people must not be reported as interacting"
    );
}

#[test]
fn test_classify_context_uses_real_scene_data_not_hardcoded_indoor() {
    let engine = ActivityRecognitionEngine::new();
    let scene = scene_with(vec![object((0.0, 0.0, 1.0, 1.0))], Vec::new());
    let context = engine
        .context_classifier
        .classify_context(&scene)
        .expect("classify_context failed");

    assert_eq!(context.scene_type, "test_scene");
    assert_ne!(context.scene_type, "indoor");
    assert_eq!(
        context.environment_factors.get("object_count").copied(),
        Some(1.0)
    );
}

#[test]
fn test_apply_temporal_smoothing_majority_votes_and_averages() {
    let history = vec![
        frame_result("walking", 0.1, vec![]),
        frame_result("walking", 0.2, vec![]),
        frame_result("walking", 0.3, vec![]),
    ];
    let current = frame_result("sitting", 0.6, vec![]);
    let smoothed =
        apply_temporal_smoothing(current, &history).expect("apply_temporal_smoothing failed");

    assert_eq!(
        smoothed.scene_summary.dominant_activity, "walking",
        "3 of 4 window entries are 'walking'"
    );
    assert!(
        (smoothed.scene_summary.energy_level - 0.3).abs() < 1e-6,
        "got {}",
        smoothed.scene_summary.energy_level
    );
}

#[test]
fn test_extract_motion_features_activates_optical_flow_on_second_call() {
    // Regression guard: `get_previous_frame()` used to always return
    // `None`, so the real Lucas-Kanade optical-flow code in
    // `compute_optical_flow` was permanently unreachable.
    let engine = ActivityRecognitionEngine::new();
    let mut frame1 = Array3::<f32>::zeros((4, 4, 1));
    let mut frame2 = Array3::<f32>::zeros((4, 4, 1));
    for y in 0..4 {
        for x in 0..4 {
            frame1[[y, x, 0]] = x as f32 * 0.1;
            frame2[[y, x, 0]] = x as f32 * 0.1 + 0.05; // uniform temporal shift
        }
    }

    let first = engine
        .extract_motion_features(&frame1.view())
        .expect("extract_motion_features failed");
    for y in 1..3 {
        for x in 1..3 {
            assert_eq!(
                first[[y, x, 0]],
                0.0,
                "no previous frame yet: flow must be zero"
            );
            assert_eq!(first[[y, x, 1]], 0.0);
        }
    }

    let second = engine
        .extract_motion_features(&frame2.view())
        .expect("extract_motion_features failed");
    let any_nonzero = (1..3)
        .any(|y| (1..3).any(|x| second[[y, x, 0]].abs() > 1e-6 || second[[y, x, 1]].abs() > 1e-6));
    assert!(
        any_nonzero,
        "optical flow must activate now that frame1 is tracked as the previous frame"
    );
}

#[test]
fn test_apply_temporal_smoothing_noop_with_empty_history() {
    let current = frame_result("sitting", 0.6, vec![]);
    let smoothed = apply_temporal_smoothing(current, &[]).expect("apply_temporal_smoothing failed");
    assert_eq!(smoothed.scene_summary.dominant_activity, "sitting");
    assert_eq!(smoothed.scene_summary.energy_level, 0.6);
}
