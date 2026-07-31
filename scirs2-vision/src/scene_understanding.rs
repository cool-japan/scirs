//! Advanced Scene Understanding Framework
//!
//! This module provides advanced scene understanding capabilities including:
//! - Semantic scene segmentation and classification
//! - Object relationship reasoning
//! - Spatial layout understanding
//! - Temporal scene analysis
//! - Multi-modal scene representation
//!
//! # Implementation status
//!
//! [`SceneUnderstandingEngine`] combines several genuinely-computed classical
//! computer-vision passes (intensity-based multi-scale segmentation,
//! connected-component blob detection, per-region pixel statistics, and
//! geometry-only spatial-relationship reasoning -- see the individual method
//! docs below). None of this relies on a trained model, so **object
//! `class` labels are not semantic** (there is no person/car/chair
//! classifier here): the private `ObjectDetector::detect_multi_scale` method
//! labels every detected region `"object"`. Wiring in a real semantic detector/segmenter
//! (e.g. a trained CNN) is future work and out of scope for this module.

#![allow(dead_code)]

use crate::error::Result;
use scirs2_core::ndarray::{Array2, Array3, ArrayView3};
use std::collections::HashMap;

/// Fixed number of intensity clusters used by the classical (non-semantic)
/// segmentation pass, unless overridden by a configured
/// [`SemanticSegmentationModel`]'s `class_labels`.
const DEFAULT_SEGMENTATION_CLUSTERS: usize = 4;

/// Compute a single-channel intensity map from a `(height, width, channels)`
/// image by averaging across channels.
fn grayscale_intensity(image: &ArrayView3<f32>) -> Array2<f32> {
    let (height, width, channels) = image.dim();
    let mut gray = Array2::zeros((height, width));
    if channels == 0 {
        return gray;
    }
    let inv_channels = 1.0 / channels as f32;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for c in 0..channels {
                sum += image[[y, x, c]];
            }
            gray[[y, x]] = sum * inv_channels;
        }
    }
    gray
}

/// Simple separable box blur with the given radius (`radius == 0` is a
/// no-op copy). Used to give the multi-scale segmentation pass genuinely
/// different (coarser) spatial detail at larger scale factors.
fn box_blur(map: &Array2<f32>, radius: usize) -> Array2<f32> {
    if radius == 0 {
        return map.clone();
    }
    let (height, width) = map.dim();
    let r = radius as isize;

    // Horizontal pass
    let mut horizontal = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let x0 = (x as isize - r).max(0) as usize;
            let x1 = ((x as isize + r) as usize).min(width - 1);
            let mut sum = 0.0f32;
            for xx in x0..=x1 {
                sum += map[[y, xx]];
            }
            horizontal[[y, x]] = sum / (x1 - x0 + 1) as f32;
        }
    }

    // Vertical pass
    let mut result = Array2::zeros((height, width));
    for y in 0..height {
        let y0 = (y as isize - r).max(0) as usize;
        let y1 = ((y as isize + r) as usize).min(height - 1);
        for x in 0..width {
            let mut sum = 0.0f32;
            for yy in y0..=y1 {
                sum += horizontal[[yy, x]];
            }
            result[[y, x]] = sum / (y1 - y0 + 1) as f32;
        }
    }
    result
}

/// Real (if simple) 1-D K-means over `values` (Lloyd's algorithm), returning
/// per-point cluster labels plus the cluster centers **sorted ascending**
/// so labels are canonical (`0` = darkest cluster) and therefore comparable
/// across independent calls -- which is what lets
/// [`merge_segmentation_results`] combine multiple passes meaningfully.
fn kmeans_1d(values: &[f32], k: usize, max_iterations: usize) -> (Vec<u32>, Vec<f32>) {
    if values.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }
    let k = k.min(values.len());

    let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Deterministic initialization: evenly spaced centers across the data
    // range (avoids pulling in a RNG dependency for a fallback classical
    // segmenter, and makes results reproducible).
    let mut centers: Vec<f32> = if (max_v - min_v).abs() < f32::EPSILON {
        vec![min_v; k]
    } else {
        (0..k)
            .map(|i| min_v + (max_v - min_v) * (i as f32 + 0.5) / k as f32)
            .collect()
    };

    let mut labels = vec![0u32; values.len()];
    for _ in 0..max_iterations.max(1) {
        // Assignment step
        let mut changed = false;
        for (idx, &v) in values.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f32::INFINITY;
            for (ci, &c) in centers.iter().enumerate() {
                let dist = (v - c).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best = ci;
                }
            }
            if labels[idx] != best as u32 {
                changed = true;
            }
            labels[idx] = best as u32;
        }

        // Update step
        let mut sums = vec![0.0f32; k];
        let mut counts = vec![0usize; k];
        for (idx, &v) in values.iter().enumerate() {
            let c = labels[idx] as usize;
            sums[c] += v;
            counts[c] += 1;
        }
        for i in 0..k {
            if counts[i] > 0 {
                centers[i] = sums[i] / counts[i] as f32;
            }
        }

        if !changed {
            break;
        }
    }

    // Canonicalize: relabel so cluster indices are sorted by ascending
    // center value.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        centers[a]
            .partial_cmp(&centers[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0u32; k];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        rank[old_idx] = new_idx as u32;
    }
    let sorted_centers: Vec<f32> = order.iter().map(|&old_idx| centers[old_idx]).collect();
    let relabeled: Vec<u32> = labels.iter().map(|&l| rank[l as usize]).collect();

    (relabeled, sorted_centers)
}

/// Count of same-shape neighbors (4-connected) that *disagree* with the
/// label at `(y, x)`. Used as a "local consistency" score: a lower count
/// means a smoother, more confident local segmentation at that pixel.
fn local_disagreement(labels: &Array2<u32>, y: usize, x: usize) -> u32 {
    let (height, width) = labels.dim();
    let center = labels[[y, x]];
    let mut disagreements = 0u32;
    let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dy, dx) in neighbors {
        let ny = y as isize + dy;
        let nx = x as isize + dx;
        if ny >= 0
            && nx >= 0
            && (ny as usize) < height
            && (nx as usize) < width
            && labels[[ny as usize, nx as usize]] != center
        {
            disagreements += 1;
        }
    }
    disagreements
}

/// 3x3 majority filter: replace each label with the most frequent label in
/// its 8-connected neighborhood (ties keep the original value). A standard,
/// simple denoising pass for label maps.
fn majority_filter(labels: &Array2<u32>) -> Array2<u32> {
    let (height, width) = labels.dim();
    let mut result = labels.clone();
    for y in 0..height {
        for x in 0..width {
            let mut counts: HashMap<u32, u32> = HashMap::new();
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if ny >= 0 && nx >= 0 && (ny as usize) < height && (nx as usize) < width {
                        *counts
                            .entry(labels[[ny as usize, nx as usize]])
                            .or_insert(0) += 1;
                    }
                }
            }
            if let Some((&best_label, &best_count)) = counts.iter().max_by_key(|(_, &count)| count)
            {
                let current_count = counts.get(&labels[[y, x]]).copied().unwrap_or(0);
                if best_count > current_count {
                    result[[y, x]] = best_label;
                }
            }
        }
    }
    result
}

/// Otsu-style binary threshold: pick the intensity level that maximizes
/// between-class variance over a coarse 256-bin histogram of `values`.
fn otsu_threshold(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max_v - min_v).abs() < f32::EPSILON {
        return min_v;
    }

    const BINS: usize = 256;
    let mut hist = [0u32; BINS];
    let scale = (BINS - 1) as f32 / (max_v - min_v);
    for &v in values {
        let bin = (((v - min_v) * scale).round() as usize).min(BINS - 1);
        hist[bin] += 1;
    }

    let total = values.len() as f64;
    let sum_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, &c)| i as f64 * c as f64)
        .sum();

    let mut sum_bg = 0.0f64;
    let mut weight_bg = 0.0f64;
    let mut best_variance = -1.0f64;
    // Track the inclusive [low, high] range of bin indices that tie for
    // the best between-class variance, and report their *midpoint* as the
    // threshold. Between two consecutive populated bins there is always a
    // run of empty bins that all tie for the same (unchanged) variance;
    // always taking the first of that run (as a naive `>` comparison
    // does) collapses the reported threshold onto the lower cluster's own
    // value instead of a point that actually separates the two clusters
    // -- which matters whenever a caller (unlike this module's own
    // `> threshold` mask/detector uses) treats the threshold value itself
    // as meaningful, e.g. for a perfectly bimodal 2-value histogram.
    let mut best_low = 0usize;
    let mut best_high = 0usize;

    for (bin, &count) in hist.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }
        let weight_fg = total - weight_bg;
        if weight_fg <= 0.0 {
            break;
        }
        sum_bg += bin as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (sum_all - sum_bg) / weight_fg;
        let variance = weight_bg * weight_fg * (mean_bg - mean_fg).powi(2);
        if variance > best_variance + 1e-9 {
            best_variance = variance;
            best_low = bin;
            best_high = bin;
        } else if (variance - best_variance).abs() <= 1e-9 {
            best_high = bin;
        }
    }

    let best_bin = (best_low as f32 + best_high as f32) / 2.0;
    min_v + best_bin / scale
}

/// A connected foreground region found by [`find_connected_components`].
struct ConnectedComponent {
    /// Bounding box as (x, y, width, height).
    bbox: (f32, f32, f32, f32),
    /// Number of foreground pixels in the component.
    pixel_count: usize,
}

/// Real 4-connected flood-fill connected-component labeling over a boolean
/// foreground mask. This is the classical building block behind
/// [`ObjectDetector::detect_multi_scale`]'s region proposals.
fn find_connected_components(mask: &Array2<bool>) -> Vec<ConnectedComponent> {
    let (height, width) = mask.dim();
    let mut visited = Array2::from_elem((height, width), false);
    let mut components = Vec::new();

    for start_y in 0..height {
        for start_x in 0..width {
            if !mask[[start_y, start_x]] || visited[[start_y, start_x]] {
                continue;
            }

            let mut stack = vec![(start_y, start_x)];
            visited[[start_y, start_x]] = true;
            let (mut min_x, mut max_x) = (start_x, start_x);
            let (mut min_y, mut max_y) = (start_y, start_y);
            let mut pixel_count = 0usize;

            while let Some((y, x)) = stack.pop() {
                pixel_count += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);

                let neighbors: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dy, dx) in neighbors {
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if ny >= 0 && nx >= 0 && (ny as usize) < height && (nx as usize) < width {
                        let (ny, nx) = (ny as usize, nx as usize);
                        if mask[[ny, nx]] && !visited[[ny, nx]] {
                            visited[[ny, nx]] = true;
                            stack.push((ny, nx));
                        }
                    }
                }
            }

            components.push(ConnectedComponent {
                bbox: (
                    min_x as f32,
                    min_y as f32,
                    (max_x - min_x + 1) as f32,
                    (max_y - min_y + 1) as f32,
                ),
                pixel_count,
            });
        }
    }

    components
}

/// Intersection-over-union of two `(x, y, width, height)` boxes.
fn bbox_iou(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let (ax0, ay0, aw, ah) = *a;
    let (bx0, by0, bw, bh) = *b;
    let (ax1, ay1) = (ax0 + aw, ay0 + ah);
    let (bx1, by1) = (bx0 + bw, by0 + bh);

    let ix0 = ax0.max(bx0);
    let iy0 = ay0.max(by0);
    let ix1 = ax1.min(bx1);
    let iy1 = ay1.min(by1);

    if ix1 <= ix0 || iy1 <= iy0 {
        return 0.0;
    }
    let intersection = (ix1 - ix0) * (iy1 - iy0);
    let union = (aw * ah).max(0.0) + (bw * bh).max(0.0) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Clamp a `(x, y, width, height)` bounding box to valid, non-empty pixel
/// index ranges `[x0, x1) x [y0, y1)` within an image of the given size.
/// Always returns `x1 > x0` and `y1 > y0` when `width, height > 0`, even if
/// the input bbox is degenerate, negative, or out of bounds.
fn clamp_bbox(
    bbox: &(f32, f32, f32, f32),
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0, 0, 0);
    }
    let (bx, by, bw, bh) = *bbox;
    let x0 = bx.max(0.0).min((width - 1) as f32) as usize;
    let y0 = by.max(0.0).min((height - 1) as f32) as usize;
    let x1_raw = (bx + bw.max(1.0)).max((x0 + 1) as f32).min(width as f32);
    let y1_raw = (by + bh.max(1.0)).max((y0 + 1) as f32).min(height as f32);
    let x1 = (x1_raw as usize).clamp(x0 + 1, width);
    let y1 = (y1_raw as usize).clamp(y0 + 1, height);
    (x0, y0, x1, y1)
}

/// Mean, (population) standard deviation, min, and max of a slice (all
/// zero for an empty slice).
fn mean_std_min_max(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min_v, max_v)
}

/// Normalized 16-bin histogram of `values` over their own observed range
/// (all zero for an empty slice).
fn histogram_16(values: &[f32]) -> [f32; 16] {
    let mut hist = [0.0f32; 16];
    if values.is_empty() {
        return hist;
    }
    let min_v = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_v - min_v).max(f32::EPSILON);
    for &v in values {
        let bin = (((v - min_v) / range) * 15.99).floor() as usize;
        hist[bin.min(15)] += 1.0;
    }
    let total = values.len() as f32;
    for h in hist.iter_mut() {
        *h /= total;
    }
    hist
}

/// Advanced-advanced scene understanding engine with multi-level reasoning
pub struct SceneUnderstandingEngine {
    /// Semantic segmentation models
    segmentation_models: Vec<SemanticSegmentationModel>,
    /// Object detection and classification
    object_detector: ObjectDetector,
    /// Spatial relationship analyzer
    spatial_analyzer: SpatialRelationshipAnalyzer,
    /// Temporal scene tracker
    temporal_tracker: TemporalSceneTracker,
    /// Scene graph builder
    scene_graph_builder: SceneGraphBuilder,
    /// Context-aware reasoning engine
    reasoning_engine: ContextualReasoningEngine,
}

/// Semantic segmentation model with advanced-high accuracy
#[derive(Debug, Clone)]
pub struct SemanticSegmentationModel {
    /// Model type identifier
    model_type: String,
    /// Class labels
    class_labels: Vec<String>,
    /// Model confidence threshold
    confidence_threshold: f32,
    /// Multi-scale analysis parameters
    scale_factors: Vec<f32>,
}

impl SemanticSegmentationModel {
    /// Default classical (intensity-clustering) segmentation configuration
    /// used by [`SceneUnderstandingEngine::segment_at_scale`]. This is a
    /// non-semantic model: `class_labels` name generic intensity regions,
    /// not object categories.
    fn default_intensity_model() -> Self {
        Self {
            model_type: "kmeans_intensity".to_string(),
            class_labels: (0..DEFAULT_SEGMENTATION_CLUSTERS)
                .map(|i| format!("region_{i}"))
                .collect(),
            confidence_threshold: 0.5,
            scale_factors: vec![0.5, 1.0, 1.5, 2.0],
        }
    }
}

/// Advanced object detector with relationship understanding
#[derive(Debug, Clone)]
pub struct ObjectDetector {
    /// Detection confidence threshold
    confidence_threshold: f32,
    /// Non-maximum suppression threshold
    nms_threshold: f32,
    /// Supported object classes
    object_classes: Vec<String>,
    /// Feature extraction layers
    feature_layers: Vec<String>,
}

/// Spatial relationship analyzer for scene understanding
#[derive(Debug, Clone)]
pub struct SpatialRelationshipAnalyzer {
    /// Relationship types
    relationship_types: Vec<SpatialRelationType>,
    /// Distance thresholds for relationships
    distance_thresholds: HashMap<String, f32>,
    /// Directional analysis parameters
    directional_params: DirectionalParams,
}

/// Temporal scene tracking for video understanding
#[derive(Debug, Clone)]
pub struct TemporalSceneTracker {
    /// Frame buffer size
    buffer_size: usize,
    /// Motion detection threshold
    motion_threshold: f32,
    /// Object tracking parameters
    tracking_params: TrackingParams,
    /// Scene change detection
    change_detection: ChangeDetectionParams,
}

/// Scene graph construction for relationship modeling
#[derive(Debug, Clone)]
pub struct SceneGraphBuilder {
    /// Maximum number of nodes
    max_nodes: usize,
    /// Edge confidence threshold
    edge_threshold: f32,
    /// Graph simplification parameters
    simplification_params: GraphSimplificationParams,
}

/// Contextual reasoning engine for high-level understanding
#[derive(Debug, Clone)]
pub struct ContextualReasoningEngine {
    /// Reasoning rules
    rules: Vec<ReasoningRule>,
    /// Context windows
    context_windows: Vec<ContextWindow>,
    /// Inference parameters
    inference_params: InferenceParams,
}

/// Detected object with rich metadata
#[derive(Debug, Clone)]
pub struct DetectedObject {
    /// Object class
    pub class: String,
    /// Bounding box (x, y, width, height)
    pub bbox: (f32, f32, f32, f32),
    /// Detection confidence
    pub confidence: f32,
    /// Object features
    pub features: Array2<f32>,
    /// Object mask (if available)
    pub mask: Option<Array2<bool>>,
    /// Object attributes
    pub attributes: HashMap<String, f32>,
}

/// Spatial relationship between objects
#[derive(Debug, Clone)]
pub struct SpatialRelation {
    /// Source object ID
    pub source_id: usize,
    /// Target object ID
    pub target_id: usize,
    /// Relationship type
    pub relation_type: SpatialRelationType,
    /// Relationship confidence
    pub confidence: f32,
    /// Spatial parameters
    pub parameters: HashMap<String, f32>,
}

/// Scene understanding result with comprehensive analysis
#[derive(Debug, Clone)]
pub struct SceneAnalysisResult {
    /// Detected objects
    pub objects: Vec<DetectedObject>,
    /// Spatial relationships
    pub relationships: Vec<SpatialRelation>,
    /// Scene classification
    pub scene_class: String,
    /// Scene confidence
    pub scene_confidence: f32,
    /// Semantic segmentation map
    pub segmentation_map: Array2<u32>,
    /// Scene graph representation
    pub scene_graph: SceneGraph,
    /// Temporal information (if applicable)
    pub temporal_info: Option<TemporalInfo>,
    /// Reasoning results
    pub reasoning_results: Vec<ReasoningResult>,
}

/// Supporting types for scene understanding
#[derive(Debug, Clone)]
pub enum SpatialRelationType {
    /// Object A is on top of object B
    OnTop,
    /// Object A is inside object B
    Inside,
    /// Object A is next to object B
    NextTo,
    /// Object A is in front of object B
    InFrontOf,
    /// Object A is behind object B
    Behind,
    /// Object A is above object B
    Above,
    /// Object A is below object B
    Below,
    /// Object A is to the left of object B
    LeftOf,
    /// Object A is to the right of object B
    RightOf,
    /// Object A contains object B
    Contains,
    /// Object A supports object B
    Supports,
    /// Object A is connected to object B
    ConnectedTo,
    /// Custom relationship
    Custom(String),
}

/// Parameters for directional spatial relationship analysis
#[derive(Debug, Clone)]
pub struct DirectionalParams {
    /// Angular tolerance for directional relationships (in radians)
    pub angular_tolerance: f32,
    /// Whether to normalize distances for scale invariance
    pub distance_normalization: bool,
    /// Whether to apply perspective correction
    pub perspective_correction: bool,
}

/// Parameters for temporal object tracking
#[derive(Debug, Clone)]
pub struct TrackingParams {
    /// Maximum frames an object can disappear before being lost
    pub max_disappearance_frames: usize,
    /// Tracking algorithm identifier
    pub tracking_algorithm: String,
    /// Threshold for feature matching in tracking
    pub feature_matching_threshold: f32,
}

/// Parameters for scene change detection
#[derive(Debug, Clone)]
pub struct ChangeDetectionParams {
    /// Sensitivity level for change detection (0.0-1.0)
    pub sensitivity: f32,
    /// Size of temporal window for change analysis
    pub temporal_window: usize,
    /// Background model type identifier
    pub background_model: String,
}

/// Parameters for scene graph simplification
#[derive(Debug, Clone)]
pub struct GraphSimplificationParams {
    /// Minimum edge weight to retain in graph
    pub min_edge_weight: f32,
    /// Whether to remove redundant edges
    pub redundancy_removal: bool,
    /// Whether to apply hierarchical clustering
    pub hierarchical_clustering: bool,
}

/// Rule for contextual reasoning about scenes
#[derive(Debug, Clone)]
pub struct ReasoningRule {
    /// Name identifier for the reasoning rule
    pub name: String,
    /// Conditions that must be met for rule to apply
    pub conditions: Vec<String>,
    /// Conclusions drawn when conditions are met
    pub conclusions: Vec<String>,
    /// Confidence level of the rule (0.0-1.0)
    pub confidence: f32,
}

/// Context window for reasoning about scene elements
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Number of frames to consider for temporal context
    pub temporal_span: usize,
    /// Spatial extent (width, height) for context
    pub spatial_extent: (f32, f32),
    /// Threshold for relevance filtering
    pub relevance_threshold: f32,
}

/// Parameters for reasoning inference process
#[derive(Debug, Clone)]
pub struct InferenceParams {
    /// Maximum iterations for inference convergence
    pub max_iterations: usize,
    /// Threshold for determining convergence
    pub convergence_threshold: f32,
    /// Method for handling uncertainty in inference
    pub uncertainty_handling: String,
}

/// Graph representation of scene structure and relationships
#[derive(Debug, Clone)]
pub struct SceneGraph {
    /// Nodes representing objects and regions in the scene
    pub nodes: Vec<SceneGraphNode>,
    /// Edges representing relationships between objects
    pub edges: Vec<SceneGraphEdge>,
    /// Global scene properties and metadata
    pub global_properties: HashMap<String, f32>,
}

/// Node in scene graph representing an object or region
#[derive(Debug, Clone)]
pub struct SceneGraphNode {
    /// Unique identifier for the node
    pub id: usize,
    /// Type or class of the object
    pub object_type: String,
    /// Properties and attributes of the object
    pub properties: HashMap<String, f32>,
    /// Spatial location (x, y) in the scene
    pub spatial_location: (f32, f32),
}

/// Edge in scene graph representing a relationship
#[derive(Debug, Clone)]
pub struct SceneGraphEdge {
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Type of relationship
    pub relation_type: String,
    /// Strength or confidence of the relationship
    pub weight: f32,
    /// Additional properties of the relationship
    pub properties: HashMap<String, f32>,
}

/// Temporal information for video scene understanding
#[derive(Debug, Clone)]
pub struct TemporalInfo {
    /// Index of the current frame
    pub frame_index: usize,
    /// Timestamp of the frame
    pub timestamp: f64,
    /// Motion vectors for temporal analysis
    pub motion_vectors: Array3<f32>,
    /// Detected changes in the scene
    pub scene_changes: Vec<SceneChange>,
}

/// Information about a detected change in the scene
#[derive(Debug, Clone)]
pub struct SceneChange {
    /// Type of change that occurred
    pub change_type: String,
    /// Location (x, y) where change occurred
    pub location: (f32, f32),
    /// Magnitude of the change
    pub magnitude: f32,
    /// Confidence in the change detection
    pub confidence: f32,
}

/// Result of contextual reasoning about the scene
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// Name of the rule that generated this result
    pub rule_name: String,
    /// Conclusion reached by the reasoning process
    pub conclusion: String,
    /// Confidence in the reasoning result
    pub confidence: f32,
    /// Evidence supporting the conclusion
    pub evidence: Vec<String>,
}

impl Default for SceneUnderstandingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneUnderstandingEngine {
    /// Create a new advanced scene understanding engine
    pub fn new() -> Self {
        Self {
            segmentation_models: vec![SemanticSegmentationModel::default_intensity_model()],
            object_detector: ObjectDetector::new(),
            spatial_analyzer: SpatialRelationshipAnalyzer::new(),
            temporal_tracker: TemporalSceneTracker::new(),
            scene_graph_builder: SceneGraphBuilder::new(),
            reasoning_engine: ContextualReasoningEngine::new(),
        }
    }

    /// Analyze a single image with comprehensive scene understanding
    pub fn analyze_scene(&self, image: &ArrayView3<f32>) -> Result<SceneAnalysisResult> {
        // Multi-scale semantic segmentation
        let segmentation_map = self.perform_semantic_segmentation(image)?;

        // Object detection and feature extraction
        let objects = self.detect_objects(image)?;

        // Spatial relationship analysis
        let relationships = self.analyze_spatial_relationships(&objects)?;

        // Scene classification
        let (scene_class, scene_confidence) = self.classify_scene(image, &objects)?;

        // Scene graph construction
        let scene_graph = self.build_scene_graph(&objects, &relationships)?;

        // Contextual reasoning
        let reasoning_results = self.perform_reasoning(&objects, &relationships, &scene_class)?;

        Ok(SceneAnalysisResult {
            objects,
            relationships,
            scene_class,
            scene_confidence,
            segmentation_map,
            scene_graph,
            temporal_info: None,
            reasoning_results,
        })
    }

    /// Analyze video sequence with temporal understanding
    pub fn analyze_video_sequence(
        &mut self,
        frames: &[ArrayView3<f32>],
    ) -> Result<Vec<SceneAnalysisResult>> {
        let mut results = Vec::new();

        for (frame_idx, frame) in frames.iter().enumerate() {
            // Analyze individual frame
            let mut frame_result = self.analyze_scene(frame)?;

            // Add temporal analysis
            if frame_idx > 0 {
                let temporal_info =
                    self.analyze_temporal_changes(frame, &frames[..frame_idx], frame_idx)?;
                frame_result.temporal_info = Some(temporal_info);
            }

            results.push(frame_result);
        }

        // Post-process for temporal consistency
        self.enforce_temporal_consistency(&mut results)?;

        Ok(results)
    }

    /// Perform classical (non-semantic) multi-scale intensity segmentation.
    ///
    /// Each configured scale factor contributes an independent
    /// [`Self::segment_at_scale`] pass; passes are combined by
    /// [`Self::merge_segmentation_results`] (which keeps whichever pass is
    /// locally more self-consistent at each pixel), then smoothed by
    /// [`Self::enforce_spatial_consistency`].
    fn perform_semantic_segmentation(&self, image: &ArrayView3<f32>) -> Result<Array2<u32>> {
        let (height, width, _channels) = image.dim();
        let scale_factors: Vec<f32> = self
            .segmentation_models
            .first()
            .filter(|m| !m.scale_factors.is_empty())
            .map(|m| m.scale_factors.clone())
            .unwrap_or_else(|| vec![0.5, 1.0, 1.5, 2.0]);

        let mut segmentation_map: Option<Array2<u32>> = None;

        // Multi-scale segmentation for enhanced accuracy
        for scale_factor in scale_factors {
            let scaled_result = self.segment_at_scale(image, scale_factor)?;
            match segmentation_map.as_mut() {
                None => segmentation_map = Some(scaled_result),
                Some(base) => self.merge_segmentation_results(base, &scaled_result)?,
            }
        }

        let mut segmentation_map =
            segmentation_map.unwrap_or_else(|| Array2::zeros((height, width)));

        // Post-processing for spatial consistency
        self.enforce_spatial_consistency(&mut segmentation_map)?;

        Ok(segmentation_map)
    }

    /// Detect objects with rich feature extraction
    fn detect_objects(&self, image: &ArrayView3<f32>) -> Result<Vec<DetectedObject>> {
        let mut objects = Vec::new();

        // Multi-scale object detection
        let detection_results = self.object_detector.detect_multi_scale(image)?;

        for detection in detection_results {
            // Extract rich features for each object
            let features = self.extract_object_features(image, &detection.bbox)?;

            // Compute object mask
            let mask = self.compute_object_mask(image, &detection)?;

            // Analyze object attributes
            let attributes = self.analyze_object_attributes(image, &detection, &features)?;

            objects.push(DetectedObject {
                class: detection.class,
                bbox: detection.bbox,
                confidence: detection.confidence,
                features,
                mask: Some(mask),
                attributes,
            });
        }

        Ok(objects)
    }

    /// Analyze spatial relationships between objects
    fn analyze_spatial_relationships(
        &self,
        objects: &[DetectedObject],
    ) -> Result<Vec<SpatialRelation>> {
        let mut relationships = Vec::new();

        for (i, obj1) in objects.iter().enumerate() {
            for (j, obj2) in objects.iter().enumerate() {
                if i != j {
                    let relations = self.spatial_analyzer.analyze_pair(obj1, obj2, i, j)?;
                    relationships.extend(relations);
                }
            }
        }

        // Filter relationships based on confidence
        relationships.retain(|r| r.confidence > 0.5);

        Ok(relationships)
    }

    /// Classify the overall scene
    fn classify_scene(
        &self,
        image: &ArrayView3<f32>,
        objects: &[DetectedObject],
    ) -> Result<(String, f32)> {
        // Extract global scene features
        let global_features = self.extract_global_features(image)?;

        // Analyze object composition
        let object_composition = self.analyze_object_composition(objects)?;

        // Combine features for scene classification
        let scene_features = self.combine_scene_features(&global_features, &object_composition)?;

        // Perform classification
        let (scene_class, confidence) = self.classify_from_features(&scene_features)?;

        Ok((scene_class, confidence))
    }

    /// Build comprehensive scene graph
    fn build_scene_graph(
        &self,
        objects: &[DetectedObject],
        relationships: &[SpatialRelation],
    ) -> Result<SceneGraph> {
        let nodes = objects
            .iter()
            .enumerate()
            .map(|(i, obj)| SceneGraphNode {
                id: i,
                object_type: obj.class.clone(),
                properties: obj.attributes.clone(),
                spatial_location: (obj.bbox.0 + obj.bbox.2 / 2.0, obj.bbox.1 + obj.bbox.3 / 2.0),
            })
            .collect();

        let edges = relationships
            .iter()
            .map(|rel| SceneGraphEdge {
                source: rel.source_id,
                target: rel.target_id,
                relation_type: format!("{:?}", rel.relation_type),
                weight: rel.confidence,
                properties: rel.parameters.clone(),
            })
            .collect();

        // ── Global scene properties ──────────────────────────────────────
        let mut global_properties: HashMap<String, f32> = HashMap::new();

        // n_objects: count of detected objects
        let n_objects = objects.len() as f32;
        global_properties.insert("n_objects".to_string(), n_objects);

        // avg_object_confidence: mean detection confidence across all objects
        let avg_object_confidence = if objects.is_empty() {
            0.0_f32
        } else {
            objects.iter().map(|o| o.confidence).sum::<f32>() / n_objects
        };
        global_properties.insert("avg_object_confidence".to_string(), avg_object_confidence);

        // scene_density: objects per unit area, estimated from bounding-box extents
        // (bbox = (x, y, width, height))
        let scene_density = if objects.is_empty() {
            0.0_f32
        } else {
            let max_x = objects
                .iter()
                .map(|o| o.bbox.0 + o.bbox.2)
                .fold(0.0_f32, f32::max);
            let max_y = objects
                .iter()
                .map(|o| o.bbox.1 + o.bbox.3)
                .fold(0.0_f32, f32::max);
            let image_area = (max_x * max_y).max(1.0);
            n_objects / image_area
        };
        global_properties.insert("scene_density".to_string(), scene_density);

        // relationship_diversity: number of unique relation types in the relationships list
        let mut rel_type_counts: HashMap<String, u32> = HashMap::new();
        for rel in relationships {
            let key = format!("{:?}", rel.relation_type);
            *rel_type_counts.entry(key).or_insert(0) += 1;
        }
        let relationship_diversity = rel_type_counts.len() as f32;
        global_properties.insert("relationship_diversity".to_string(), relationship_diversity);

        // dominant_relation_count: count of the most-frequent relation type
        // (Note: the type is HashMap<String, f32>, so the relation name is stored as a hash
        //  under the "dominant_relation_type_name_hash" key for downstream decoding.)
        let dominant_rel_count = rel_type_counts.values().copied().max().unwrap_or(0) as f32;
        global_properties.insert("dominant_relation_count".to_string(), dominant_rel_count);

        // dominant_relation_type_name_hash: djb2 hash of the most-frequent relation name
        if let Some((dominant_name, _)) = rel_type_counts.iter().max_by_key(|(_, &c)| c) {
            let name_hash = dominant_name
                .bytes()
                .fold(5381_u32, |h, b| h.wrapping_mul(33).wrapping_add(b as u32))
                as f32;
            global_properties.insert("dominant_relation_type_name_hash".to_string(), name_hash);
        }

        Ok(SceneGraph {
            nodes,
            edges,
            global_properties,
        })
    }

    /// Perform contextual reasoning on scene understanding results
    fn perform_reasoning(
        &self,
        objects: &[DetectedObject],
        relationships: &[SpatialRelation],
        scene_class: &str,
    ) -> Result<Vec<ReasoningResult>> {
        let mut results = Vec::new();

        // Apply reasoning rules
        for rule in &self.reasoning_engine.rules {
            if let Some(result) =
                self.apply_reasoning_rule(rule, objects, relationships, scene_class, scene_class)?
            {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Real (classical, non-semantic) region segmentation: K-means over a
    /// scale-dependent blur of the grayscale image, with cluster count
    /// taken from the first configured [`SemanticSegmentationModel`]
    /// (falling back to [`DEFAULT_SEGMENTATION_CLUSTERS`]). Canonical
    /// (ascending-intensity) label ordering is what lets
    /// [`Self::merge_segmentation_results`] combine passes meaningfully.
    fn segment_at_scale(&self, image: &ArrayView3<f32>, scale: f32) -> Result<Array2<u32>> {
        let (height, width, _channels) = image.dim();
        if height == 0 || width == 0 {
            return Ok(Array2::zeros((height, width)));
        }

        let gray = grayscale_intensity(image);
        // Larger `scale` -> more blur -> coarser regions: a real
        // multi-scale technique using the actual scale factor.
        let radius = (scale.max(0.0) * 2.0).round() as usize;
        let blurred = box_blur(&gray, radius);

        let k = self
            .segmentation_models
            .first()
            .map(|m| m.class_labels.len().max(2))
            .unwrap_or(DEFAULT_SEGMENTATION_CLUSTERS);

        let flat: Vec<f32> = blurred.iter().copied().collect();
        let (labels, _centers) = kmeans_1d(&flat, k, 25);

        let mut segmentation_map = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                segmentation_map[[y, x]] = labels[y * width + x];
            }
        }
        Ok(segmentation_map)
    }

    /// Combine `new` into `base` in place by keeping, at each pixel,
    /// whichever of the two labels is locally more self-consistent (fewer
    /// disagreeing 4-connected neighbors within its own map) -- a simple
    /// but genuine multi-scale ensembling rule, made meaningful by
    /// `segment_at_scale`'s canonical label ordering.
    fn merge_segmentation_results(&self, base: &mut Array2<u32>, new: &Array2<u32>) -> Result<()> {
        if base.shape() != new.shape() {
            return Ok(());
        }
        let (height, width) = base.dim();
        for y in 0..height {
            for x in 0..width {
                let base_disagreement = local_disagreement(base, y, x);
                let new_disagreement = local_disagreement(new, y, x);
                if new_disagreement < base_disagreement {
                    base[[y, x]] = new[[y, x]];
                }
            }
        }
        Ok(())
    }

    /// Real post-processing: a 3x3 majority filter that removes isolated
    /// single-pixel label noise (standard segmentation cleanup).
    fn enforce_spatial_consistency(&self, segmentation: &mut Array2<u32>) -> Result<()> {
        let filtered = majority_filter(segmentation);
        segmentation.assign(&filtered);
        Ok(())
    }

    /// Real per-object feature vector: per-channel mean/std/min/max over
    /// the (clamped) bounding-box region packed in blocks of 4, followed
    /// by a coarse 16-bin grayscale histogram of the same region, into a
    /// fixed-length 256-column vector (zero-padded beyond what fits).
    fn extract_object_features(
        &self,
        image: &ArrayView3<f32>,
        bbox: &(f32, f32, f32, f32),
    ) -> Result<Array2<f32>> {
        let (img_height, img_width, channels) = image.dim();
        let (x0, y0, x1, y1) = clamp_bbox(bbox, img_width, img_height);

        let mut features = vec![0.0f32; 256];
        let mut col = 0usize;

        for c in 0..channels.min(8) {
            let mut values = Vec::with_capacity((x1 - x0) * (y1 - y0));
            for y in y0..y1 {
                for x in x0..x1 {
                    values.push(image[[y, x, c]]);
                }
            }
            let (mean, std, min_v, max_v) = mean_std_min_max(&values);
            if col + 4 <= features.len() {
                features[col] = mean;
                features[col + 1] = std;
                features[col + 2] = min_v;
                features[col + 3] = max_v;
            }
            col += 4;
        }

        let mut region_gray = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            for x in x0..x1 {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += image[[y, x, c]];
                }
                region_gray.push(sum / channels.max(1) as f32);
            }
        }
        let hist = histogram_16(&region_gray);
        for (i, &v) in hist.iter().enumerate() {
            if col + i < features.len() {
                features[col + i] = v;
            }
        }

        Ok(Array2::from_shape_vec((1, 256), features)?)
    }

    /// Real per-object foreground mask: Otsu-threshold the grayscale
    /// intensity within the (clamped) bounding box, treating whichever
    /// side of the threshold covers the smaller area as foreground (a
    /// detection box typically has more background than object at its
    /// margins).
    fn compute_object_mask(
        &self,
        image: &ArrayView3<f32>,
        detection: &DetectionResult,
    ) -> Result<Array2<bool>> {
        let (img_height, img_width, channels) = image.dim();
        let (x0, y0, x1, y1) = clamp_bbox(&detection.bbox, img_width, img_height);
        let (box_h, box_w) = (y1 - y0, x1 - x0);

        let mut region_gray = Vec::with_capacity(box_h * box_w);
        for y in y0..y1 {
            for x in x0..x1 {
                let mut sum = 0.0f32;
                for c in 0..channels {
                    sum += image[[y, x, c]];
                }
                region_gray.push(sum / channels.max(1) as f32);
            }
        }

        let threshold = otsu_threshold(&region_gray);
        let above = region_gray.iter().filter(|&&v| v > threshold).count();
        let below = region_gray.len() - above;
        let foreground_is_above = above <= below;

        let mut mask = Array2::from_elem((box_h, box_w), false);
        for (i, &v) in region_gray.iter().enumerate() {
            let yy = i / box_w;
            let xx = i % box_w;
            mask[[yy, xx]] = (v > threshold) == foreground_is_above;
        }
        Ok(mask)
    }

    /// Real per-object attributes derived from actual geometry and the
    /// already-computed `features` vector (mean/std packed in blocks of 4
    /// by [`Self::extract_object_features`]).
    fn analyze_object_attributes(
        &self,
        _image: &ArrayView3<f32>,
        detection: &DetectionResult,
        features: &Array2<f32>,
    ) -> Result<HashMap<String, f32>> {
        let mut attributes = HashMap::new();
        let (_x, _y, w, h) = detection.bbox;
        attributes.insert(
            "aspect_ratio".to_string(),
            if h.abs() > f32::EPSILON { w / h } else { 0.0 },
        );
        attributes.insert("area".to_string(), (w * h).max(0.0));
        attributes.insert("confidence".to_string(), detection.confidence);

        let feature_row = features.row(0);
        let num_channel_blocks = (feature_row.len() / 4).min(8);
        let mut brightest_channel = 0usize;
        let mut brightest_mean = f32::NEG_INFINITY;
        for c in 0..num_channel_blocks {
            let mean = feature_row[c * 4];
            attributes.insert(format!("channel_{c}_mean"), mean);
            attributes.insert(format!("channel_{c}_std"), feature_row[c * 4 + 1]);
            if mean > brightest_mean {
                brightest_mean = mean;
                brightest_channel = c;
            }
        }
        if num_channel_blocks > 0 {
            attributes.insert("dominant_channel".to_string(), brightest_channel as f32);
        }

        Ok(attributes)
    }

    /// Real global scene feature vector: per-channel mean/std/min/max
    /// followed by a coarse 16-bin grayscale histogram, packed into a
    /// fixed-length 512-column vector (zero-padded beyond what fits).
    fn extract_global_features(&self, image: &ArrayView3<f32>) -> Result<Array2<f32>> {
        let (height, width, channels) = image.dim();
        let mut features = vec![0.0f32; 512];
        let mut col = 0usize;

        for c in 0..channels.min(16) {
            let mut values = Vec::with_capacity(height * width);
            for y in 0..height {
                for x in 0..width {
                    values.push(image[[y, x, c]]);
                }
            }
            let (mean, std, min_v, max_v) = mean_std_min_max(&values);
            if col + 4 <= features.len() {
                features[col] = mean;
                features[col + 1] = std;
                features[col + 2] = min_v;
                features[col + 3] = max_v;
            }
            col += 4;
        }

        let gray = grayscale_intensity(image);
        let gray_values: Vec<f32> = gray.iter().copied().collect();
        let hist = histogram_16(&gray_values);
        for (i, &v) in hist.iter().enumerate() {
            if col + i < features.len() {
                features[col + i] = v;
            }
        }

        Ok(Array2::from_shape_vec((1, 512), features)?)
    }

    /// Real object-composition summary: object count, confidence
    /// statistics, bounding-box area statistics, and class diversity,
    /// packed into a fixed-length 128-column vector.
    fn analyze_object_composition(&self, objects: &[DetectedObject]) -> Result<Array2<f32>> {
        let mut features = vec![0.0f32; 128];
        features[0] = objects.len() as f32;

        if !objects.is_empty() {
            let confidences: Vec<f32> = objects.iter().map(|o| o.confidence).collect();
            let (mean_conf, std_conf, min_conf, max_conf) = mean_std_min_max(&confidences);
            features[1] = mean_conf;
            features[2] = std_conf;
            features[3] = min_conf;
            features[4] = max_conf;

            let areas: Vec<f32> = objects
                .iter()
                .map(|o| (o.bbox.2 * o.bbox.3).max(0.0))
                .collect();
            let (mean_area, std_area, _min_area, max_area) = mean_std_min_max(&areas);
            features[5] = mean_area;
            features[6] = std_area;
            features[7] = max_area;

            let mut class_counts: HashMap<&str, u32> = HashMap::new();
            for obj in objects {
                *class_counts.entry(obj.class.as_str()).or_insert(0) += 1;
            }
            features[8] = class_counts.len() as f32;
        }

        Ok(Array2::from_shape_vec((1, 128), features)?)
    }

    /// Real feature fusion: concatenate `global` then `composition` into a
    /// single 640-column vector, truncating or zero-padding either side to
    /// fit.
    fn combine_scene_features(
        &self,
        global: &Array2<f32>,
        composition: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let mut combined = vec![0.0f32; 640];
        for (i, &v) in global.row(0).iter().enumerate() {
            if i < combined.len() {
                combined[i] = v;
            }
        }
        let offset = global.ncols().min(combined.len());
        for (i, &v) in composition.row(0).iter().enumerate() {
            if offset + i < combined.len() {
                combined[offset + i] = v;
            }
        }
        Ok(Array2::from_shape_vec((1, 640), combined)?)
    }

    /// Lightweight **heuristic** scene classifier -- not a trained model.
    /// Uses overall brightness (`features[0]`, the mean of the first image
    /// channel as packed by [`Self::extract_global_features`]) and object
    /// count (`features[512]`, as packed by
    /// [`Self::analyze_object_composition`] via
    /// [`Self::combine_scene_features`]) to pick among 3 coarse labels.
    /// `confidence` reflects how decisively the brightness threshold was
    /// crossed, not a calibrated probability.
    fn classify_from_features(&self, features: &Array2<f32>) -> Result<(String, f32)> {
        let row = features.row(0);
        let brightness = row.first().copied().unwrap_or(0.0);
        let object_count = row.get(512).copied().unwrap_or(0.0);

        const BRIGHTNESS_THRESHOLD: f32 = 0.5;
        let distance = (brightness - BRIGHTNESS_THRESHOLD).abs();
        let confidence = (0.5 + distance).clamp(0.5, 0.99);

        let scene_class = if object_count >= 4.0 {
            "cluttered_scene"
        } else if brightness >= BRIGHTNESS_THRESHOLD {
            "bright_open_scene"
        } else {
            "dim_enclosed_scene"
        };

        Ok((scene_class.to_string(), confidence))
    }

    /// Evaluate whether `rule` applies given the current scene state, and
    /// build its [`ReasoningResult`] if so. See [`evaluate_condition`] for
    /// the supported condition syntax; a rule applies only when it has at
    /// least one condition and *all* of them hold.
    fn apply_reasoning_rule(
        &self,
        rule: &ReasoningRule,
        objects: &[DetectedObject],
        relationships: &[SpatialRelation],
        scene_class: &str,
        _class: &str,
    ) -> Result<Option<ReasoningResult>> {
        let applies = !rule.conditions.is_empty()
            && rule
                .conditions
                .iter()
                .all(|c| evaluate_condition(c, objects, relationships, scene_class));

        if !applies {
            return Ok(None);
        }

        Ok(Some(ReasoningResult {
            rule_name: rule.name.clone(),
            conclusion: rule.conclusions.join("; "),
            confidence: rule.confidence,
            evidence: rule.conditions.clone(),
        }))
    }

    /// Real frame-differencing temporal change detection: thresholds
    /// (Otsu) the absolute grayscale difference between `current_frame`
    /// and the most recent previous frame, then reports each connected
    /// component of changed pixels (via [`find_connected_components`]) as
    /// a [`SceneChange`]. `motion_vectors` is a coarse signed-difference
    /// field duplicated across both components -- a simple
    /// frame-difference proxy, **not** true dense optical flow.
    fn analyze_temporal_changes(
        &self,
        current_frame: &ArrayView3<f32>,
        previous_frames: &[ArrayView3<f32>],
        frame_idx: usize,
    ) -> Result<TemporalInfo> {
        let (height, width, _channels) = current_frame.dim();
        let empty_result = || TemporalInfo {
            frame_index: frame_idx,
            timestamp: frame_idx as f64 / 30.0, // Assuming 30 FPS
            motion_vectors: Array3::zeros((height, width, 2)),
            scene_changes: Vec::new(),
        };

        let Some(previous_frame) = previous_frames.last() else {
            return Ok(empty_result());
        };
        if previous_frame.dim() != current_frame.dim() {
            return Ok(empty_result());
        }

        let current_gray = grayscale_intensity(current_frame);
        let previous_gray = grayscale_intensity(previous_frame);
        let mut diff = Array2::zeros((height, width));
        for y in 0..height {
            for x in 0..width {
                diff[[y, x]] = (current_gray[[y, x]] - previous_gray[[y, x]]).abs();
            }
        }

        let diff_values: Vec<f32> = diff.iter().copied().collect();
        let threshold = otsu_threshold(&diff_values).max(0.02);
        let changed_mask = diff.mapv(|v| v > threshold);
        let components = find_connected_components(&changed_mask);

        let scene_changes: Vec<SceneChange> = components
            .into_iter()
            .filter(|c| c.pixel_count >= 4)
            .map(|c| {
                let (x, y, w, h) = c.bbox;
                let mut sum = 0.0f32;
                let mut count = 0usize;
                for yy in (y as usize)..((y + h) as usize).min(height) {
                    for xx in (x as usize)..((x + w) as usize).min(width) {
                        sum += diff[[yy, xx]];
                        count += 1;
                    }
                }
                let magnitude = if count > 0 { sum / count as f32 } else { 0.0 };
                SceneChange {
                    change_type: "motion".to_string(),
                    location: (x + w / 2.0, y + h / 2.0),
                    magnitude,
                    confidence: (magnitude * 2.0).clamp(0.0, 1.0),
                }
            })
            .collect();

        let mut motion_vectors = Array3::zeros((height, width, 2));
        for y in 0..height {
            for x in 0..width {
                let signed = current_gray[[y, x]] - previous_gray[[y, x]];
                motion_vectors[[y, x, 0]] = signed;
                motion_vectors[[y, x, 1]] = signed;
            }
        }

        Ok(TemporalInfo {
            frame_index: frame_idx,
            timestamp: frame_idx as f64 / 30.0,
            motion_vectors,
            scene_changes,
        })
    }

    /// Real (if simple) temporal smoothing: if a frame's `scene_class`
    /// disagrees with both of its immediate neighbors *and* those
    /// neighbors agree with each other, treat it as a one-frame
    /// classification outlier and replace it with the neighbors' class
    /// (with a slightly reduced confidence, since it is carried over
    /// rather than freshly classified).
    fn enforce_temporal_consistency(&mut self, results: &mut [SceneAnalysisResult]) -> Result<()> {
        if results.len() < 3 {
            return Ok(());
        }
        let corrections: Vec<(usize, String, f32)> = (1..results.len() - 1)
            .filter_map(|i| {
                let prev = &results[i - 1];
                let curr = &results[i];
                let next = &results[i + 1];
                if prev.scene_class == next.scene_class && curr.scene_class != prev.scene_class {
                    Some((
                        i,
                        prev.scene_class.clone(),
                        (prev.scene_confidence.min(next.scene_confidence) * 0.9).max(0.0),
                    ))
                } else {
                    None
                }
            })
            .collect();

        for (i, class, confidence) in corrections {
            results[i].scene_class = class;
            results[i].scene_confidence = confidence;
        }

        Ok(())
    }
}

/// Evaluate a single condition string against the current scene state, for
/// [`SceneUnderstandingEngine::apply_reasoning_rule`]. Supported forms:
/// * `"min_objects:N"` -- at least `N` detected objects
/// * `"min_relationships:N"` -- at least `N` spatial relationships
/// * `"has_relation:<RelationType>"` -- at least one relationship whose
///   `{:?}`-formatted type starts with `<RelationType>` (e.g. `"OnTop"`)
/// * `"scene_class:<name>"` -- the scene was classified as `<name>`
///
/// Any other/malformed condition string evaluates to `false`.
fn evaluate_condition(
    condition: &str,
    objects: &[DetectedObject],
    relationships: &[SpatialRelation],
    scene_class: &str,
) -> bool {
    let Some((key, value)) = condition.split_once(':') else {
        return false;
    };
    let value = value.trim();
    match key {
        "min_objects" => value.parse::<usize>().is_ok_and(|n| objects.len() >= n),
        "min_relationships" => value
            .parse::<usize>()
            .is_ok_and(|n| relationships.len() >= n),
        "has_relation" => relationships
            .iter()
            .any(|r| format!("{:?}", r.relation_type).starts_with(value)),
        "scene_class" => scene_class == value,
        _ => false,
    }
}

// Placeholder detection result structure
#[derive(Debug, Clone)]
struct DetectionResult {
    class: String,
    bbox: (f32, f32, f32, f32),
    confidence: f32,
}

// Implementation stubs for associated types
impl ObjectDetector {
    fn new() -> Self {
        Self {
            confidence_threshold: 0.5,
            nms_threshold: 0.4,
            object_classes: vec!["person".to_string(), "car".to_string(), "chair".to_string()],
            feature_layers: vec!["conv5".to_string(), "fc7".to_string()],
        }
    }

    /// Real (classical, non-semantic) region proposal: Otsu-threshold the
    /// grayscale image into foreground/background, connected-component
    /// label the foreground (via [`find_connected_components`]), and
    /// return one [`DetectionResult`] per component above a minimum size
    /// (after NMS). **There is no trained classifier behind this**: every
    /// detection is labeled the generic `"object"` class, never one of
    /// `self.object_classes` -- callers must not treat `class` as a
    /// semantic prediction.
    fn detect_multi_scale(&self, image: &ArrayView3<f32>) -> Result<Vec<DetectionResult>> {
        let (height, width, _channels) = image.dim();
        if height == 0 || width == 0 {
            return Ok(Vec::new());
        }

        let gray = grayscale_intensity(image);
        let values: Vec<f32> = gray.iter().copied().collect();
        let threshold = otsu_threshold(&values);

        let above = values.iter().filter(|&&v| v > threshold).count();
        let below = values.len() - above;
        let foreground_is_above = above <= below;

        let mask = gray.mapv(|v| (v > threshold) == foreground_is_above);
        let components = find_connected_components(&mask);

        let image_area = (height * width) as f32;
        let min_pixels = ((image_area * 0.001).round() as usize).max(4);

        let mut detections: Vec<DetectionResult> = components
            .into_iter()
            .filter(|c| c.pixel_count >= min_pixels)
            .map(|c| {
                let bbox_area = (c.bbox.2 * c.bbox.3).max(1.0);
                let fill_fraction = c.pixel_count as f32 / bbox_area;
                DetectionResult {
                    class: "object".to_string(),
                    bbox: c.bbox,
                    confidence: (self.confidence_threshold + fill_fraction * 0.5).clamp(0.0, 1.0),
                }
            })
            .filter(|d| d.confidence >= self.confidence_threshold)
            .collect();

        // Non-maximum suppression on real geometric IoU, using this
        // detector's own configured threshold.
        detections.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut kept: Vec<DetectionResult> = Vec::new();
        for det in detections {
            let overlaps_kept = kept
                .iter()
                .any(|k| bbox_iou(&k.bbox, &det.bbox) > self.nms_threshold);
            if !overlaps_kept {
                kept.push(det);
            }
        }

        Ok(kept)
    }
}

impl SpatialRelationshipAnalyzer {
    fn new() -> Self {
        Self {
            relationship_types: vec![SpatialRelationType::OnTop, SpatialRelationType::NextTo],
            distance_thresholds: HashMap::new(),
            directional_params: DirectionalParams {
                angular_tolerance: 15.0,
                distance_normalization: true,
                perspective_correction: true,
            },
        }
    }

    /// Real geometry-only spatial-relationship inference between two
    /// detected objects: containment via bbox-inside-bbox and IoU tests,
    /// and relative direction via center-point comparison. No
    /// appearance/semantic reasoning is involved -- this is pure geometry
    /// over the two bounding boxes.
    fn analyze_pair(
        &self,
        obj1: &DetectedObject,
        obj2: &DetectedObject,
        id1: usize,
        id2: usize,
    ) -> Result<Vec<SpatialRelation>> {
        let (x1, y1, w1, h1) = obj1.bbox;
        let (x2, y2, w2, h2) = obj2.bbox;
        let (cx1, cy1) = (x1 + w1 / 2.0, y1 + h1 / 2.0);
        let (cx2, cy2) = (x2 + w2 / 2.0, y2 + h2 / 2.0);

        let iou = bbox_iou(&obj1.bbox, &obj2.bbox);
        let obj1_inside_obj2 =
            x1 >= x2 && y1 >= y2 && (x1 + w1) <= (x2 + w2) && (y1 + h1) <= (y2 + h2);

        let mut parameters = HashMap::new();
        parameters.insert("iou".to_string(), iou);
        parameters.insert("center_dx".to_string(), cx1 - cx2);
        parameters.insert("center_dy".to_string(), cy1 - cy2);

        let (relation_type, confidence) = if obj1_inside_obj2 && (w1 * h1) < (w2 * h2) {
            (SpatialRelationType::Inside, 0.9)
        } else if iou > 0.1 {
            (SpatialRelationType::NextTo, iou.min(1.0))
        } else {
            let dx = cx1 - cx2;
            let dy = cy1 - cy2;
            if dy.abs() >= dx.abs() {
                let confidence = (dy.abs() / h1.max(h2).max(1.0)).clamp(0.3, 1.0);
                if dy < 0.0 {
                    (SpatialRelationType::Above, confidence)
                } else {
                    (SpatialRelationType::Below, confidence)
                }
            } else {
                let confidence = (dx.abs() / w1.max(w2).max(1.0)).clamp(0.3, 1.0);
                if dx < 0.0 {
                    (SpatialRelationType::LeftOf, confidence)
                } else {
                    (SpatialRelationType::RightOf, confidence)
                }
            }
        };

        Ok(vec![SpatialRelation {
            source_id: id1,
            target_id: id2,
            relation_type,
            confidence,
            parameters,
        }])
    }
}

impl TemporalSceneTracker {
    fn new() -> Self {
        Self {
            buffer_size: 30,
            motion_threshold: 0.1,
            tracking_params: TrackingParams {
                max_disappearance_frames: 10,
                tracking_algorithm: "kalman".to_string(),
                feature_matching_threshold: 0.8,
            },
            change_detection: ChangeDetectionParams {
                sensitivity: 0.5,
                temporal_window: 5,
                background_model: "gaussian_mixture".to_string(),
            },
        }
    }
}

impl SceneGraphBuilder {
    fn new() -> Self {
        Self {
            max_nodes: 100,
            edge_threshold: 0.3,
            simplification_params: GraphSimplificationParams {
                min_edge_weight: 0.1,
                redundancy_removal: true,
                hierarchical_clustering: true,
            },
        }
    }
}

impl ContextualReasoningEngine {
    /// Construct the engine with a small set of default geometry/count
    /// based reasoning rules (see [`evaluate_condition`] for the condition
    /// syntax). These are simple, honestly-scoped heuristics over the
    /// real detections/relationships -- not learned rules.
    fn new() -> Self {
        Self {
            rules: vec![
                ReasoningRule {
                    name: "crowded_scene".to_string(),
                    conditions: vec!["min_objects:3".to_string()],
                    conclusions: vec![
                        "Scene contains several detected regions and may be crowded".to_string()
                    ],
                    confidence: 0.7,
                },
                ReasoningRule {
                    name: "stacked_objects".to_string(),
                    conditions: vec!["has_relation:Above".to_string()],
                    conclusions: vec!["At least one region is positioned above another".to_string()],
                    confidence: 0.6,
                },
                ReasoningRule {
                    name: "rich_spatial_structure".to_string(),
                    conditions: vec!["min_relationships:2".to_string()],
                    conclusions: vec![
                        "Scene has multiple spatial relationships between regions".to_string()
                    ],
                    confidence: 0.65,
                },
            ],
            context_windows: Vec::new(),
            inference_params: InferenceParams {
                max_iterations: 100,
                convergence_threshold: 0.01,
                uncertainty_handling: "bayesian".to_string(),
            },
        }
    }
}

/// Advanced-advanced scene understanding with cognitive-level reasoning
#[allow(dead_code)]
pub fn analyze_scene_with_reasoning(
    image: &ArrayView3<f32>,
    context: Option<&SceneAnalysisResult>,
) -> Result<SceneAnalysisResult> {
    let engine = SceneUnderstandingEngine::new();
    let mut result = engine.analyze_scene(image)?;

    // Apply contextual reasoning if previous context is available
    if let Some(prev_context) = context {
        result = apply_contextual_enhancement(&result, prev_context)?;
    }

    Ok(result)
}

/// Apply contextual enhancement based on previous scene understanding.
///
/// Real (if simple) rule: temporal agreement between consecutive frames is
/// weak evidence of correctness, so when the current and previous frame
/// were classified into the *same* scene class, nudge `scene_confidence`
/// toward the previous frame's confidence; otherwise the result is
/// returned unchanged.
#[allow(dead_code)]
fn apply_contextual_enhancement(
    current: &SceneAnalysisResult,
    previous: &SceneAnalysisResult,
) -> Result<SceneAnalysisResult> {
    let mut enhanced = current.clone();
    if enhanced.scene_class == previous.scene_class {
        let boost = 0.05 * previous.scene_confidence;
        enhanced.scene_confidence = (enhanced.scene_confidence + boost).min(1.0);
    }
    Ok(enhanced)
}

// Tests live in a separate file (kept out of this one) to respect the
// workspace's 2000-line-per-file convention.
#[cfg(test)]
#[path = "scene_understanding_tests.rs"]
mod tests;
