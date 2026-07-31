//! Tests for `scene_understanding.rs`.
//!
//! Kept in a separate file (included via `#[path]`) to respect the
//! workspace's 2000-line-per-file convention. All test images are
//! non-constant/non-trivial so a fabricating stub (e.g. one that always
//! returns the same hardcoded output regardless of input) cannot pass.

use super::*;
use scirs2_core::ndarray::Array3;

/// Build a `(height, width, 3)` RGB-ish image, uniformly `background`
/// everywhere except for a `size x size` square of `foreground` placed at
/// `(top, left)`.
fn image_with_bright_square(
    height: usize,
    width: usize,
    background: f32,
    top: usize,
    left: usize,
    size: usize,
    foreground: f32,
) -> Array3<f32> {
    Array3::from_shape_fn((height, width, 3), |(y, x, _c)| {
        if y >= top && y < top + size && x >= left && x < left + size {
            foreground
        } else {
            background
        }
    })
}

fn uniform_image(height: usize, width: usize, value: f32) -> Array3<f32> {
    Array3::from_elem((height, width, 3), value)
}

fn detected_object(bbox: (f32, f32, f32, f32), confidence: f32) -> DetectedObject {
    DetectedObject {
        class: "object".to_string(),
        bbox,
        confidence,
        features: Array2::zeros((1, 4)),
        mask: None,
        attributes: HashMap::new(),
    }
}

// ---------------------------------------------------------------------
// Pure-function helpers
// ---------------------------------------------------------------------

#[test]
fn test_grayscale_intensity_averages_channels() {
    let image = Array3::from_shape_vec((1, 1, 3), vec![0.2_f32, 0.4, 0.6]).expect("valid shape");
    let gray = grayscale_intensity(&image.view());
    assert!((gray[[0, 0]] - 0.4).abs() < 1e-6);
}

#[test]
fn test_kmeans_1d_separates_two_well_separated_clusters() {
    let values = vec![0.0_f32, 0.1, 0.05, 0.02, 10.0, 10.2, 9.9, 10.1];
    let (labels, centers) = kmeans_1d(&values, 2, 25);

    assert_eq!(centers.len(), 2);
    // Canonical ordering: center 0 must be the darker (smaller) cluster.
    assert!(centers[0] < centers[1]);
    assert!(centers[0] < 1.0);
    assert!(centers[1] > 9.0);

    // The first 4 (low) values must share one label, the last 4 (high)
    // values the other, and the two groups must differ.
    let low_label = labels[0];
    let high_label = labels[4];
    assert_ne!(low_label, high_label);
    for &l in &labels[0..4] {
        assert_eq!(l, low_label);
    }
    for &l in &labels[4..8] {
        assert_eq!(l, high_label);
    }
}

#[test]
fn test_kmeans_1d_empty_and_zero_k() {
    let (labels, centers) = kmeans_1d(&[], 3, 10);
    assert!(labels.is_empty());
    assert!(centers.is_empty());

    let (labels, centers) = kmeans_1d(&[1.0, 2.0], 0, 10);
    assert!(labels.is_empty());
    assert!(centers.is_empty());
}

#[test]
fn test_otsu_threshold_splits_bimodal_data() {
    let mut values = vec![0.0_f32; 20];
    values.extend(vec![1.0_f32; 20]);
    let threshold = otsu_threshold(&values);
    assert!(
        threshold > 0.1 && threshold < 0.9,
        "threshold {threshold} should fall between the two modes"
    );
}

#[test]
fn test_otsu_threshold_constant_data() {
    let values = vec![0.5_f32; 10];
    let threshold = otsu_threshold(&values);
    assert!((threshold - 0.5).abs() < 1e-5);
}

#[test]
fn test_find_connected_components_counts_two_separate_blobs() {
    let (h, w) = (10, 10);
    let mut mask = Array2::from_elem((h, w), false);
    // Blob 1: 2x2 at (0,0)-(1,1)
    mask[[0, 0]] = true;
    mask[[0, 1]] = true;
    mask[[1, 0]] = true;
    mask[[1, 1]] = true;
    // Blob 2: 2x2 at (5,5)-(6,6), far from blob 1 (not 4-connected to it)
    mask[[5, 5]] = true;
    mask[[5, 6]] = true;
    mask[[6, 5]] = true;
    mask[[6, 6]] = true;

    let components = find_connected_components(&mask);
    assert_eq!(components.len(), 2);
    for c in &components {
        assert_eq!(c.pixel_count, 4);
        assert_eq!(c.bbox.2, 2.0); // width
        assert_eq!(c.bbox.3, 2.0); // height
    }
}

#[test]
fn test_bbox_iou_known_value() {
    let a = (0.0_f32, 0.0, 10.0, 10.0);
    let b = (5.0_f32, 5.0, 10.0, 10.0);
    // intersection = 5x5 = 25, union = 100+100-25 = 175
    let expected = 25.0 / 175.0;
    assert!((bbox_iou(&a, &b) - expected).abs() < 1e-5);
}

#[test]
fn test_bbox_iou_no_overlap_is_zero() {
    let a = (0.0_f32, 0.0, 10.0, 10.0);
    let b = (100.0_f32, 100.0, 10.0, 10.0);
    assert_eq!(bbox_iou(&a, &b), 0.0);
}

#[test]
fn test_clamp_bbox_handles_out_of_range() {
    let (x0, y0, x1, y1) = clamp_bbox(&(1000.0, 1000.0, 50.0, 50.0), 20, 20);
    assert!(x1 > x0);
    assert!(y1 > y0);
    assert!(x1 <= 20);
    assert!(y1 <= 20);

    let (x0, y0, x1, y1) = clamp_bbox(&(-50.0, -50.0, 5.0, 5.0), 20, 20);
    assert_eq!(x0, 0);
    assert_eq!(y0, 0);
    assert!(x1 > x0 && y1 > y0);
}

#[test]
fn test_mean_std_min_max() {
    let (mean, std, min_v, max_v) = mean_std_min_max(&[1.0, 2.0, 3.0, 4.0]);
    assert!((mean - 2.5).abs() < 1e-5);
    assert!(std > 0.0);
    assert_eq!(min_v, 1.0);
    assert_eq!(max_v, 4.0);

    let (mean, std, min_v, max_v) = mean_std_min_max(&[]);
    assert_eq!((mean, std, min_v, max_v), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn test_histogram_16_sums_to_one() {
    let values: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
    let hist = histogram_16(&values);
    let total: f32 = hist.iter().sum();
    assert!((total - 1.0).abs() < 1e-5);
    // Roughly-uniform data should populate every bin.
    assert!(hist.iter().all(|&v| v > 0.0));
}

// ---------------------------------------------------------------------
// SpatialRelationshipAnalyzer::analyze_pair (pure geometry)
// ---------------------------------------------------------------------

#[test]
fn test_analyze_pair_detects_inside() {
    let engine = SceneUnderstandingEngine::new();
    let outer = detected_object((0.0, 0.0, 100.0, 100.0), 0.9);
    let inner = detected_object((10.0, 10.0, 5.0, 5.0), 0.9);

    let relations = engine
        .spatial_analyzer
        .analyze_pair(&inner, &outer, 0, 1)
        .expect("analyze_pair failed");
    assert_eq!(relations.len(), 1);
    assert!(matches!(
        relations[0].relation_type,
        SpatialRelationType::Inside
    ));
}

#[test]
fn test_analyze_pair_detects_above_below() {
    let engine = SceneUnderstandingEngine::new();
    // `a` sits higher on the image (smaller y) than `b`, with no overlap.
    let a = detected_object((0.0, 0.0, 10.0, 10.0), 0.9);
    let b = detected_object((0.0, 50.0, 10.0, 10.0), 0.9);

    let a_vs_b = engine
        .spatial_analyzer
        .analyze_pair(&a, &b, 0, 1)
        .expect("analyze_pair failed");
    assert!(matches!(
        a_vs_b[0].relation_type,
        SpatialRelationType::Above
    ));

    let b_vs_a = engine
        .spatial_analyzer
        .analyze_pair(&b, &a, 1, 0)
        .expect("analyze_pair failed");
    assert!(matches!(
        b_vs_a[0].relation_type,
        SpatialRelationType::Below
    ));
}

#[test]
fn test_analyze_pair_detects_left_right() {
    let engine = SceneUnderstandingEngine::new();
    let left = detected_object((0.0, 0.0, 10.0, 10.0), 0.9);
    let right = detected_object((80.0, 0.0, 10.0, 10.0), 0.9);

    let left_vs_right = engine
        .spatial_analyzer
        .analyze_pair(&left, &right, 0, 1)
        .expect("analyze_pair failed");
    assert!(matches!(
        left_vs_right[0].relation_type,
        SpatialRelationType::LeftOf
    ));
}

// ---------------------------------------------------------------------
// evaluate_condition
// ---------------------------------------------------------------------

#[test]
fn test_evaluate_condition_forms() {
    let objects = vec![
        detected_object((0.0, 0.0, 1.0, 1.0), 0.9),
        detected_object((0.0, 0.0, 1.0, 1.0), 0.9),
        detected_object((0.0, 0.0, 1.0, 1.0), 0.9),
    ];
    let relationships = vec![SpatialRelation {
        source_id: 0,
        target_id: 1,
        relation_type: SpatialRelationType::Above,
        confidence: 0.8,
        parameters: HashMap::new(),
    }];

    assert!(evaluate_condition(
        "min_objects:3",
        &objects,
        &relationships,
        "x"
    ));
    assert!(!evaluate_condition(
        "min_objects:4",
        &objects,
        &relationships,
        "x"
    ));
    assert!(evaluate_condition(
        "min_relationships:1",
        &objects,
        &relationships,
        "x"
    ));
    assert!(evaluate_condition(
        "has_relation:Above",
        &objects,
        &relationships,
        "x"
    ));
    assert!(!evaluate_condition(
        "has_relation:Below",
        &objects,
        &relationships,
        "x"
    ));
    assert!(evaluate_condition(
        "scene_class:cluttered_scene",
        &objects,
        &relationships,
        "cluttered_scene"
    ));
    assert!(!evaluate_condition(
        "malformed",
        &objects,
        &relationships,
        "x"
    ));
}

// ---------------------------------------------------------------------
// classify_from_features: heuristic, but genuinely responds to content
// ---------------------------------------------------------------------

#[test]
fn test_classify_from_features_responds_to_brightness_and_count() {
    let engine = SceneUnderstandingEngine::new();

    let mut bright_few = vec![0.0f32; 640];
    bright_few[0] = 0.9; // bright
    bright_few[512] = 0.0; // no objects
    let (bright_class, _) = engine
        .classify_from_features(&Array2::from_shape_vec((1, 640), bright_few).expect("shape"))
        .expect("classify_from_features failed");

    let mut dim_few = vec![0.0f32; 640];
    dim_few[0] = 0.1; // dim
    dim_few[512] = 0.0;
    let (dim_class, _) = engine
        .classify_from_features(&Array2::from_shape_vec((1, 640), dim_few).expect("shape"))
        .expect("classify_from_features failed");

    let mut many_objects = vec![0.0f32; 640];
    many_objects[0] = 0.9;
    many_objects[512] = 10.0; // many objects
    let (crowded_class, _) = engine
        .classify_from_features(&Array2::from_shape_vec((1, 640), many_objects).expect("shape"))
        .expect("classify_from_features failed");

    // Different inputs must yield different labels -- a hardcoded stub
    // (always "indoor_scene", as the original code did) would fail this.
    assert_ne!(bright_class, dim_class);
    assert_ne!(bright_class, crowded_class);
    assert_eq!(bright_class, "bright_open_scene");
    assert_eq!(dim_class, "dim_enclosed_scene");
    assert_eq!(crowded_class, "cluttered_scene");
}

// ---------------------------------------------------------------------
// merge_segmentation_results: prefers the locally-consistent map
// ---------------------------------------------------------------------

#[test]
fn test_merge_segmentation_results_prefers_locally_consistent_map() {
    let engine = SceneUnderstandingEngine::new();

    // `smooth`: all zero except the very center (locally consistent almost
    // everywhere). `noisy`: checkerboard (locally inconsistent everywhere).
    let mut smooth = Array2::<u32>::zeros((5, 5));
    smooth[[2, 2]] = 1;
    let noisy = Array2::from_shape_fn((5, 5), |(y, x)| ((y + x) % 2) as u32);

    let mut base = smooth.clone();
    engine
        .merge_segmentation_results(&mut base, &noisy)
        .expect("merge failed");

    // At a corner pixel (0,0), `smooth` has zero disagreements with its
    // neighbors while `noisy` (checkerboard) has the maximum possible --
    // the merge must keep `smooth`'s value there.
    assert_eq!(base[[0, 0]], smooth[[0, 0]]);
}

// ---------------------------------------------------------------------
// compute_object_mask: real Otsu-based foreground extraction
// ---------------------------------------------------------------------

#[test]
fn test_compute_object_mask_finds_bright_foreground() {
    let engine = SceneUnderstandingEngine::new();
    // A 10x10 region, dark background with a 4x4 bright block in the
    // middle -- the bright block is the minority, so it should be
    // extracted as foreground.
    let image = image_with_bright_square(10, 10, 0.1, 3, 3, 4, 0.9);
    let detection = DetectionResult {
        class: "object".to_string(),
        bbox: (0.0, 0.0, 10.0, 10.0),
        confidence: 0.8,
    };

    let mask = engine
        .compute_object_mask(&image.view(), &detection)
        .expect("compute_object_mask failed");

    let true_count = mask.iter().filter(|&&v| v).count();
    // 16 bright pixels are the true foreground; allow some slack for
    // Otsu's exact threshold placement.
    assert!(
        (10..=22).contains(&true_count),
        "expected roughly 16 foreground pixels, got {true_count}"
    );
    // The center of the bright block must be marked foreground.
    assert!(mask[[5, 5]]);
    // A corner (background) must not be.
    assert!(!mask[[0, 0]]);
}

// ---------------------------------------------------------------------
// End-to-end: SceneUnderstandingEngine::analyze_scene
// ---------------------------------------------------------------------

#[test]
fn test_analyze_scene_detects_bright_square_and_segments_content() {
    let engine = SceneUnderstandingEngine::new();
    let image = image_with_bright_square(40, 40, 0.1, 5, 5, 12, 0.9);

    let result = engine
        .analyze_scene(&image.view())
        .expect("analyze_scene failed");

    // A stub that always returns an empty object list (as the original
    // `detect_multi_scale` did) fails this.
    assert!(
        !result.objects.is_empty(),
        "expected the bright square to be detected as a real object"
    );

    // The detected bbox should roughly overlap the true square location
    // (5,5)-(17,17).
    let true_bbox = (5.0, 5.0, 12.0, 12.0);
    let best_iou = result
        .objects
        .iter()
        .map(|o| bbox_iou(&o.bbox, &true_bbox))
        .fold(0.0_f32, f32::max);
    assert!(
        best_iou > 0.3,
        "best detected bbox should overlap the true square, got IoU {best_iou}"
    );

    // Segmentation must reflect real content: a two-intensity-level image
    // should produce at least 2 distinct labels, not an all-zero map.
    let mut unique_labels: Vec<u32> = result.segmentation_map.iter().copied().collect();
    unique_labels.sort_unstable();
    unique_labels.dedup();
    assert!(
        unique_labels.len() >= 2,
        "expected at least 2 distinct segmentation labels, got {unique_labels:?}"
    );

    // Scene graph should have one node per detected object.
    assert_eq!(result.scene_graph.nodes.len(), result.objects.len());
}

#[test]
fn test_analyze_scene_differs_for_different_content() {
    let engine = SceneUnderstandingEngine::new();

    let with_square = image_with_bright_square(40, 40, 0.1, 5, 5, 12, 0.9);
    let plain = uniform_image(40, 40, 0.1);

    let result_square = engine
        .analyze_scene(&with_square.view())
        .expect("analyze_scene failed");
    let result_plain = engine
        .analyze_scene(&plain.view())
        .expect("analyze_scene failed");

    // A uniform image has no salient region to separate from the
    // background, so it should not produce the same (non-empty) object
    // detections as the image with a real bright square -- proving the
    // pipeline responds to actual content rather than fabricating a fixed
    // answer.
    assert_ne!(result_square.objects.len(), result_plain.objects.len());
}

#[test]
fn test_analyze_scene_populates_reasoning_results_when_conditions_hold() {
    let engine = SceneUnderstandingEngine::new();
    // Four bright squares scattered across the image: enough detections
    // to trigger the "crowded_scene" default rule (`min_objects:3`).
    let mut image = uniform_image(60, 60, 0.05);
    for &(top, left) in &[(2, 2), (2, 40), (40, 2), (40, 40)] {
        for y in top..top + 8 {
            for x in left..left + 8 {
                for c in 0..3 {
                    image[[y, x, c]] = 0.95;
                }
            }
        }
    }

    let result = engine
        .analyze_scene(&image.view())
        .expect("analyze_scene failed");
    assert!(
        result.objects.len() >= 3,
        "expected at least 3 detected regions, got {}",
        result.objects.len()
    );
    assert!(
        result
            .reasoning_results
            .iter()
            .any(|r| r.rule_name == "crowded_scene"),
        "expected the crowded_scene rule to fire; reasoning_results = {:?}",
        result.reasoning_results
    );
}

#[test]
fn test_analyze_video_sequence_populates_temporal_info_and_changes() {
    let mut engine = SceneUnderstandingEngine::new();

    let frame0 = image_with_bright_square(30, 30, 0.1, 5, 5, 6, 0.9);
    // frame1: the bright square moved -- real motion between frames.
    let frame1 = image_with_bright_square(30, 30, 0.1, 15, 15, 6, 0.9);

    let frames = vec![frame0.view(), frame1.view()];
    let results = engine
        .analyze_video_sequence(&frames)
        .expect("analyze_video_sequence failed");

    assert_eq!(results.len(), 2);
    assert!(results[0].temporal_info.is_none());
    let temporal = results[1]
        .temporal_info
        .as_ref()
        .expect("frame 1 should have temporal_info");
    assert!(
        !temporal.scene_changes.is_empty(),
        "moving the square between frames should register as a scene change"
    );
}

#[test]
fn test_apply_contextual_enhancement_boosts_confidence_on_agreement() {
    let base = SceneAnalysisResult {
        objects: Vec::new(),
        relationships: Vec::new(),
        scene_class: "bright_open_scene".to_string(),
        scene_confidence: 0.6,
        segmentation_map: Array2::zeros((2, 2)),
        scene_graph: SceneGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            global_properties: HashMap::new(),
        },
        temporal_info: None,
        reasoning_results: Vec::new(),
    };
    let mut agreeing = base.clone();
    agreeing.scene_confidence = 0.6;
    let mut disagreeing = base.clone();
    disagreeing.scene_class = "dim_enclosed_scene".to_string();

    let enhanced_agree =
        apply_contextual_enhancement(&agreeing, &base).expect("enhancement failed");
    assert!(enhanced_agree.scene_confidence > agreeing.scene_confidence);

    let enhanced_disagree =
        apply_contextual_enhancement(&disagreeing, &base).expect("enhancement failed");
    assert_eq!(
        enhanced_disagree.scene_confidence,
        disagreeing.scene_confidence
    );
}
