//! Plain data types for [`super`] (the activity-recognition engine and its
//! result structures). Split out of the former monolithic
//! `activity_recognition.rs` to keep files under the workspace's line-count
//! policy; behavior is unchanged, only file layout.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

/// Advanced-advanced activity_ recognition engine with multi-level analysis
pub struct ActivityRecognitionEngine {
    /// Action detection modules
    pub(super) action_detectors: Vec<ActionDetector>,
    /// Activity sequence analyzer
    pub(super) sequence_analyzer: ActivitySequenceAnalyzer,
    /// Multi-person interaction recognizer
    pub(super) interaction_recognizer: MultiPersonInteractionRecognizer,
    /// Context-aware activity_ classifier
    pub(super) context_classifier: ContextAwareActivityClassifier,
    /// Temporal activity_ modeler
    pub(super) temporal_modeler: TemporalActivityModeler,
    /// Hierarchical activity_ decomposer
    pub(super) hierarchical_decomposer: HierarchicalActivityDecomposer,
    /// Activity knowledge base
    pub(super) knowledge_base: ActivityKnowledgeBase,
    /// Grayscale intensity channel of the most recently processed frame,
    /// used by [`Self::extract_motion_features`] to compute real
    /// frame-to-frame optical flow. `&self`-taking methods update this via
    /// interior mutability so callers don't need `&mut self` per frame.
    pub(super) previous_frame: std::cell::RefCell<Option<Array3<f32>>>,
}

/// Action detection with advanced-high precision
#[derive(Debug, Clone)]
pub struct ActionDetector {
    /// Detector name
    pub(super) name: String,
    /// Supported action types
    pub(super) action_types: Vec<String>,
    /// Detection confidence threshold
    pub(super) confidence_threshold: f32,
    /// Temporal window for action detection
    pub(super) temporal_window: usize,
    /// Feature extraction method
    pub(super) feature_method: String,
}

/// Activity sequence analysis for understanding complex behaviors
#[derive(Debug, Clone)]
pub struct ActivitySequenceAnalyzer {
    /// Maximum sequence length
    pub(super) max_sequence_length: usize,
    /// Sequence pattern models
    pub(super) pattern_models: Vec<SequencePattern>,
    /// Transition probabilities
    pub(super) transition_models: HashMap<String, TransitionModel>,
    /// Anomaly detection parameters
    pub(super) anomaly_params: AnomalyDetectionParams,
}

/// Multi-person interaction recognition
#[derive(Debug, Clone)]
pub struct MultiPersonInteractionRecognizer {
    /// Interaction types
    pub(super) interaction_types: Vec<InteractionType>,
    /// Person tracking parameters
    pub(super) tracking_params: PersonTrackingParams,
    /// Social distance modeling
    pub(super) social_distance_model: SocialDistanceModel,
    /// Group activity_ recognition
    pub(super) group_recognition: GroupActivityRecognition,
}

/// Context-aware activity_ classification
#[derive(Debug, Clone)]
pub struct ContextAwareActivityClassifier {
    /// Context features
    pub(super) context_features: Vec<ContextFeature>,
    /// Environment classifiers
    pub(super) environment_classifiers: Vec<EnvironmentClassifier>,
    /// Object-activity_ associations
    pub(super) object_associations: HashMap<String, Vec<String>>,
    /// Scene-activity_ correlations
    pub(super) scene_correlations: HashMap<String, ActivityDistribution>,
}

/// Temporal activity_ modeling for understanding dynamics
#[derive(Debug, Clone)]
pub struct TemporalActivityModeler {
    /// Temporal resolution
    pub(super) temporal_resolution: f32,
    /// Memory length for temporal modeling
    pub(super) memory_length: usize,
    /// Recurrent neural network parameters
    pub(super) rnn_params: RNNParameters,
    /// Attention mechanisms
    pub(super) attention_mechanisms: Vec<TemporalAttention>,
}

/// Hierarchical activity_ decomposition
#[derive(Debug, Clone)]
pub struct HierarchicalActivityDecomposer {
    /// Activity hierarchy levels
    pub(super) hierarchy_levels: Vec<ActivityLevel>,
    /// Decomposition rules
    pub(super) decomposition_rules: Vec<DecompositionRule>,
    /// Composition rules for building complex activities
    pub(super) composition_rules: Vec<CompositionRule>,
}

/// Activity knowledge base for reasoning
#[derive(Debug, Clone)]
pub struct ActivityKnowledgeBase {
    /// Activity definitions
    pub(super) activity_definitions: HashMap<String, ActivityDefinition>,
    /// Activity ontology
    pub(super) ontology: ActivityOntology,
    /// Common activity_ patterns
    pub(super) common_patterns: Vec<ActivityPattern>,
    /// Cultural activity_ variations
    pub(super) cultural_variations: HashMap<String, Vec<ActivityVariation>>,
}

/// Comprehensive activity_ recognition result
#[derive(Debug, Clone)]
pub struct ActivityRecognitionResult {
    /// Detected activities
    pub activities: Vec<DetectedActivity>,
    /// Activity sequences
    pub sequences: Vec<ActivitySequence>,
    /// Person interactions
    pub interactions: Vec<PersonInteraction>,
    /// Overall scene activity_ summary
    pub scene_summary: ActivitySummary,
    /// Temporal activity_ timeline
    pub timeline: ActivityTimeline,
    /// Confidence scores
    pub confidence_scores: ConfidenceScores,
    /// Uncertainty quantification
    pub uncertainty: ActivityUncertainty,
}

/// Detected activity_ with rich metadata
#[derive(Debug, Clone)]
pub struct DetectedActivity {
    /// Activity class
    pub activity_class: String,
    /// Activity subtype
    pub subtype: Option<String>,
    /// Confidence score
    pub confidence: f32,
    /// Temporal bounds (start, end)
    pub temporal_bounds: (f32, f32),
    /// Spatial region
    pub spatial_region: Option<(f32, f32, f32, f32)>,
    /// Involved persons
    pub involved_persons: Vec<PersonID>,
    /// Involved objects
    pub involved_objects: Vec<ObjectID>,
    /// Activity attributes
    pub attributes: HashMap<String, f32>,
    /// Motion characteristics
    pub motion_characteristics: MotionCharacteristics,
}

/// Activity sequence representing complex behavior chains
#[derive(Debug, Clone)]
pub struct ActivitySequence {
    /// Sequence ID
    pub sequence_id: String,
    /// Component activities
    pub activities: Vec<DetectedActivity>,
    /// Sequence type
    pub sequence_type: String,
    /// Sequence confidence
    pub confidence: f32,
    /// Transition probabilities
    pub transitions: Vec<ActivityTransition>,
    /// Sequence completeness
    pub completeness: f32,
}

/// Person interaction recognition
#[derive(Debug, Clone)]
pub struct PersonInteraction {
    /// Interaction type
    pub interaction_type: String,
    /// Participating persons
    pub participants: Vec<PersonID>,
    /// Interaction strength
    pub strength: f32,
    /// Duration
    pub duration: f32,
    /// Spatial proximity
    pub proximity: f32,
    /// Interaction attributes
    pub attributes: HashMap<String, f32>,
}

/// Overall activity_ summary for the scene
#[derive(Debug, Clone)]
pub struct ActivitySummary {
    /// Dominant activity_
    pub dominant_activity: String,
    /// Activity diversity index
    pub diversity_index: f32,
    /// Energy level of the scene
    pub energy_level: f32,
    /// Social interaction level
    pub social_interaction_level: f32,
    /// Activity complexity score
    pub complexity_score: f32,
    /// Unusual activity_ indicators
    pub anomaly_indicators: Vec<AnomalyIndicator>,
}

/// Temporal activity_ timeline
#[derive(Debug, Clone)]
pub struct ActivityTimeline {
    /// Timeline segments
    pub segments: Vec<TimelineSegment>,
    /// Timeline resolution
    pub resolution: f32,
    /// Activity flow patterns
    pub flow_patterns: Vec<FlowPattern>,
}

/// Confidence scores for different aspects
#[derive(Debug, Clone)]
pub struct ConfidenceScores {
    /// Overall recognition confidence
    pub overall: f32,
    /// Per-activity_ confidences
    pub per_activity: HashMap<String, f32>,
    /// Temporal segmentation confidence
    pub temporal_segmentation: f32,
    /// Spatial localization confidence
    pub spatial_localization: f32,
}

/// Uncertainty quantification for activity_ recognition
#[derive(Debug, Clone)]
pub struct ActivityUncertainty {
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic: f32,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric: f32,
    /// Temporal uncertainty
    pub temporal: f32,
    /// Spatial uncertainty
    pub spatial: f32,
    /// Class confusion matrix
    pub confusion_matrix: Array2<f32>,
}

// Supporting types for activity_ recognition
/// Unique identifier for a person in the scene
pub type PersonID = String;
/// Unique identifier for an object in the scene
pub type ObjectID = String;

/// Motion characteristics of detected activities
#[derive(Debug, Clone)]
pub struct MotionCharacteristics {
    /// Velocity of the motion
    pub velocity: f32,
    /// Acceleration of the motion
    pub acceleration: f32,
    /// Direction of the motion in radians
    pub direction: f32,
    /// Smoothness score of the motion
    pub smoothness: f32,
    /// Periodicity measure of the motion
    pub periodicity: f32,
}

/// Transition between activities
#[derive(Debug, Clone)]
pub struct ActivityTransition {
    /// Source activity_ name
    pub from_activity: String,
    /// Target activity_ name
    pub to_activity: String,
    /// Transition probability
    pub probability: f32,
    /// Typical duration of the transition
    pub typical_duration: f32,
}

/// Indicator of anomalous behavior
#[derive(Debug, Clone)]
pub struct AnomalyIndicator {
    /// Type of anomaly detected
    pub anomaly_type: String,
    /// Severity level of the anomaly
    pub severity: f32,
    /// Description of the anomaly
    pub description: String,
    /// Temporal location of the anomaly
    pub temporal_location: f32,
}

/// Timeline segment representing a period of activity_
#[derive(Debug, Clone)]
pub struct TimelineSegment {
    /// Start time of the segment
    pub start_time: f32,
    /// End time of the segment
    pub end_time: f32,
    /// Dominant activity_ in this segment
    pub dominant_activity: String,
    /// Mix of activities and their proportions
    pub activity_mix: HashMap<String, f32>,
}

/// Flow pattern in activity_ analysis
#[derive(Debug, Clone)]
pub struct FlowPattern {
    /// Type of flow pattern
    pub pattern_type: String,
    /// Frequency of the pattern
    pub frequency: f32,
    /// Amplitude of the pattern
    pub amplitude: f32,
    /// Phase offset of the pattern
    pub phase: f32,
}

#[derive(Debug, Clone)]
pub struct SequencePattern {
    pub pattern_name: String,
    pub activity_sequence: Vec<String>,
    pub temporal_constraints: Vec<TemporalConstraint>,
    pub occurrence_probability: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalConstraint {
    pub constraint_type: String,
    pub min_duration: f32,
    pub max_duration: f32,
    pub typical_duration: f32,
}

#[derive(Debug, Clone)]
pub struct TransitionModel {
    pub source_activity: String,
    pub transition_probabilities: HashMap<String, f32>,
    pub typical_durations: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct AnomalyDetectionParams {
    pub detection_threshold: f32,
    pub temporal_window: usize,
    pub feature_importance: Array1<f32>,
    pub novelty_detection: bool,
}

#[derive(Debug, Clone)]
pub enum InteractionType {
    Conversation,
    Collaboration,
    Competition,
    Following,
    Avoiding,
    Playing,
    Fighting,
    Helping,
    Teaching,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct PersonTrackingParams {
    pub max_tracking_distance: f32,
    pub identity_confidence_threshold: f32,
    pub re_identification_enabled: bool,
    pub track_merge_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct SocialDistanceModel {
    pub personal_space_radius: f32,
    pub social_space_radius: f32,
    pub public_space_radius: f32,
    pub cultural_factors: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct GroupActivityRecognition {
    pub min_group_size: usize,
    pub max_group_size: usize,
    pub cohesion_threshold: f32,
    pub activity_synchronization: bool,
}

#[derive(Debug, Clone)]
pub enum ContextFeature {
    SceneType,
    TimeOfDay,
    Weather,
    CrowdDensity,
    NoiseLevel,
    LightingConditions,
    ObjectPresence(String),
}

#[derive(Debug, Clone)]
pub struct EnvironmentClassifier {
    pub environment_type: String,
    pub typical_activities: Vec<String>,
    pub activity_probabilities: HashMap<String, f32>,
    pub contextual_cues: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ActivityDistribution {
    pub activities: HashMap<String, f32>,
    pub temporal_patterns: HashMap<String, TemporalPattern>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_type: String,
    pub peak_times: Vec<f32>,
    pub duration_distribution: Array1<f32>,
    pub seasonality: Option<SeasonalityInfo>,
}

#[derive(Debug, Clone)]
pub struct SeasonalityInfo {
    pub period: f32,
    pub amplitude: f32,
    pub phase_shift: f32,
}

#[derive(Debug, Clone)]
pub struct RNNParameters {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout_rate: f32,
    pub bidirectional: bool,
}

#[derive(Debug, Clone)]
pub struct TemporalAttention {
    pub attention_type: String,
    pub window_size: usize,
    pub attention_weights: Array2<f32>,
    pub learnable: bool,
}

#[derive(Debug, Clone)]
pub struct ActivityLevel {
    pub level_name: String,
    pub granularity: f32,
    pub typical_duration: f32,
    pub complexity: f32,
}

#[derive(Debug, Clone)]
pub struct DecompositionRule {
    pub rule_name: String,
    pub parent_activity: String,
    pub child_activities: Vec<String>,
    pub decomposition_conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompositionRule {
    pub rule_name: String,
    pub component_activities: Vec<String>,
    pub composite_activity: String,
    pub composition_conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ActivityDefinition {
    pub activity_name: String,
    pub description: String,
    pub typical_duration: f32,
    pub required_objects: Vec<String>,
    pub typical_poses: Vec<String>,
    pub motion_patterns: Vec<String>,
    pub contextual_requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ActivityOntology {
    pub activity_hierarchy: HashMap<String, Vec<String>>,
    pub activity_relationships: Vec<ActivityRelationship>,
    pub semantic_similarity: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ActivityRelationship {
    pub source_activity: String,
    pub target_activity: String,
    pub relationship_type: String,
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct ActivityPattern {
    pub pattern_name: String,
    pub activity_sequence: Vec<String>,
    pub temporal_structure: TemporalStructure,
    pub context_requirements: Vec<String>,
    pub occurrence_frequency: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalStructure {
    pub sequence_type: String,
    pub timing_constraints: Vec<TimingConstraint>,
    pub overlap_patterns: Vec<OverlapPattern>,
}

#[derive(Debug, Clone)]
pub struct TimingConstraint {
    pub constraint_type: String,
    pub activity_pair: (String, String),
    pub min_delay: f32,
    pub max_delay: f32,
}

#[derive(Debug, Clone)]
pub struct OverlapPattern {
    pub activity_pair: (String, String),
    pub overlap_type: String,
    pub typical_overlap: f32,
}

#[derive(Debug, Clone)]
pub struct ActivityVariation {
    pub variation_name: String,
    pub base_activity: String,
    pub cultural_context: String,
    pub modifications: HashMap<String, String>,
    pub prevalence: f32,
}

// Placeholder/support structures used by the engine's helper methods.
// Placeholder structures for compilation
#[derive(Debug, Clone)]
pub struct ContextClassification {
    pub scene_type: String,
    pub environment_factors: HashMap<String, f32>,
    pub temporal_context: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalActivityStructure {
    pub levels: Vec<ActivityLevel>,
    pub activity_tree: ActivityTree,
    pub decomposition_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ActivityTree {
    pub root: ActivityNode,
    pub nodes: Vec<ActivityNode>,
    pub edges: Vec<ActivityEdge>,
}

#[derive(Debug, Clone)]
pub struct ActivityNode {
    pub node_id: String,
    pub activity_type: String,
    pub level: usize,
    pub children: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ActivityEdge {
    pub parent: String,
    pub child: String,
    pub relationship_type: String,
}

#[derive(Debug, Clone)]
pub struct ActivityPrediction {
    pub predicted_activity: String,
    pub probability: f32,
    pub expected_start_time: f32,
    pub expected_duration: f32,
    pub confidence_interval: (f32, f32),
}
