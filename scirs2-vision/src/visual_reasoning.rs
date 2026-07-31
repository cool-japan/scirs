//! Advanced Visual Reasoning Framework
//!
//! This module provides sophisticated visual reasoning capabilities including:
//! - Causal relationship inference
//! - Visual question answering
//! - Analogical reasoning
//! - Temporal event understanding
//! - Abstract concept recognition
//! - Multi-modal reasoning integration
//!
//! # Implementation status
//!
//! This module is **experimental**. There is no trained model behind it, so
//! nothing here is genuine open-ended visual question answering, learned
//! analogical mapping, or learned causal inference. What *is* computed for
//! real, from the actual (non-semantic) detections in
//! [`crate::scene_understanding`]:
//!
//! - Answer-formatting paths like `VisualReasoningEngine::reason_what_is_happening`
//!   genuinely summarize real detections.
//! - `VisualReasoningEngine::generate_causal_explanations` surfaces the
//!   real rule-based conclusions [`crate::scene_understanding`] already
//!   computed (or honestly says none fired); it does not perform new causal
//!   inference.
//! - `VisualReasoningEngine::predict_future_events` is a real (heuristic)
//!   object-count trend read across the supplied temporal context, not
//!   genuine event prediction.
//! - `VisualReasoningEngine::analyze_causal_structure` reports the real
//!   spatial relationships already detected, as *candidate* (not confirmed)
//!   causal structure.
//! - `AnalogicalReasoningEngine::find_analogy` (via
//!   [`VisualReasoningEngine::find_analogies`]) is a real, classical
//!   structural-similarity comparison (object-class/count/relationship
//!   overlap), not learned analogical mapping.
//! - Confidence/uncertainty aggregation
//!   (`VisualReasoningEngine::estimate_overall_confidence`,
//!   `VisualReasoningEngine::quantify_uncertainty`'s `confidence_interval`
//!   and `sensitivity_analysis`) are real statistics over the underlying
//!   step/evidence values.
//!
//! Still an honest placeholder, pending either a trained model or a
//! dedicated follow-up: [`VisualReasoningEngine::infer_causality`]'s
//! temporal-pattern extraction and causal-graph construction/inference
//! (`extract_temporal_patterns`/`build_causal_graph`/`infer_effects`),
//! [`VisualReasoningEngine::recognize_abstract_concepts`] (always empty),
//! and `quantify_uncertainty`'s `epistemic_uncertainty`/
//! `aleatoric_uncertainty` split (a principled model-vs-data decomposition
//! needs a trained/probabilistic model this crate does not have). Treat any
//! output from those specific paths with caution until they are
//! implemented.

#![allow(dead_code)]
#![allow(missing_docs)]

use crate::error::Result;
use crate::scene_understanding::SceneAnalysisResult;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Advanced-advanced visual reasoning engine with cognitive-level capabilities
pub struct VisualReasoningEngine {
    /// Causal inference module
    causal_inference: CausalInferenceModule,
    /// Visual question answering system
    vqa_system: VisualQuestionAnsweringSystem,
    /// Analogical reasoning engine
    analogical_reasoning: AnalogicalReasoningEngine,
    /// Temporal event analyzer
    temporal_analyzer: TemporalEventAnalyzer,
    /// Abstract concept recognizer
    concept_recognizer: AbstractConceptRecognizer,
    /// Multi-modal integration hub
    multimodal_hub: MultiModalIntegrationHub,
    /// Knowledge base for reasoning
    knowledge_base: VisualKnowledgeBase,
}

/// Causal inference module for understanding cause-effect relationships
#[derive(Debug, Clone)]
pub struct CausalInferenceModule {
    /// Causal models
    causal_models: Vec<CausalModel>,
    /// Intervention analysis parameters
    intervention_params: InterventionParams,
    /// Counterfactual reasoning settings
    counterfactual_params: CounterfactualParams,
}

/// Visual Question Answering system with advanced reasoning
#[derive(Debug, Clone)]
pub struct VisualQuestionAnsweringSystem {
    /// Question types supported
    question_types: Vec<QuestionType>,
    /// Answer generation strategies
    answer_strategies: Vec<AnswerStrategy>,
    /// Attention mechanisms
    attention_mechanisms: Vec<AttentionMechanism>,
}

/// Analogical reasoning for pattern recognition and transfer learning
#[derive(Debug, Clone)]
pub struct AnalogicalReasoningEngine {
    /// Analogy templates
    analogy_templates: Vec<AnalogyTemplate>,
    /// Similarity metrics
    similarity_metrics: Vec<SimilarityMetric>,
    /// Transfer learning parameters
    transfer_params: TransferLearningParams,
}

/// Temporal event analysis for understanding sequences and changes
#[derive(Debug, Clone)]
pub struct TemporalEventAnalyzer {
    /// Event detection models
    event_detectors: Vec<EventDetector>,
    /// Temporal relationship models
    temporal_models: Vec<TemporalModel>,
    /// Sequence analysis parameters
    sequence_params: SequenceAnalysisParams,
}

/// Abstract concept recognition for high-level understanding
#[derive(Debug, Clone)]
pub struct AbstractConceptRecognizer {
    /// Concept hierarchies
    concept_hierarchies: Vec<ConceptHierarchy>,
    /// Feature abstraction layers
    abstraction_layers: Vec<AbstractionLayer>,
    /// Concept learning parameters
    learning_params: ConceptLearningParams,
}

/// Multi-modal integration for combining visual and other modalities
#[derive(Debug, Clone)]
pub struct MultiModalIntegrationHub {
    /// Supported modalities
    modalities: Vec<Modality>,
    /// Fusion strategies
    fusion_strategies: Vec<FusionStrategy>,
    /// Cross-modal attention mechanisms
    cross_attention: Vec<CrossModalAttention>,
}

/// Visual knowledge base for storing and retrieving reasoning knowledge
#[derive(Debug, Clone)]
pub struct VisualKnowledgeBase {
    /// Factual knowledge
    facts: HashMap<String, VisualFact>,
    /// Rules and constraints
    rules: Vec<ReasoningRule>,
    /// Concept ontology
    ontology: ConceptOntology,
}

/// Visual reasoning query for asking complex questions
#[derive(Debug, Clone)]
pub struct VisualReasoningQuery {
    /// Query type
    pub query_type: QueryType,
    /// Natural language question
    pub question: String,
    /// Query parameters
    pub parameters: HashMap<String, QueryParameter>,
    /// Context requirements
    pub context_requirements: Vec<ContextRequirement>,
}

/// Comprehensive visual reasoning result
#[derive(Debug, Clone)]
pub struct VisualReasoningResult {
    /// Answer to the query
    pub answer: ReasoningAnswer,
    /// Reasoning steps taken
    pub reasoning_steps: Vec<ReasoningStep>,
    /// Confidence in the answer
    pub confidence: f32,
    /// Evidence supporting the answer
    pub evidence: Vec<Evidence>,
    /// Alternative hypotheses considered
    pub alternatives: Vec<AlternativeHypothesis>,
    /// Uncertainty quantification
    pub uncertainty: UncertaintyQuantification,
}

/// Supporting types for visual reasoning
#[derive(Debug, Clone)]
pub enum QueryType {
    /// What is happening in the image?
    WhatIsHappening,
    /// Why is this happening?
    WhyIsHappening,
    /// What will happen next?
    WhatWillHappenNext,
    /// How are objects related?
    HowAreObjectsRelated,
    /// What if scenario analysis
    WhatIfScenario,
    /// Counting and quantification
    CountingQuery,
    /// Comparison between scenes
    ComparisonQuery,
    /// Abstract concept queries
    AbstractConceptQuery,
    /// Temporal sequence queries
    TemporalSequenceQuery,
    /// Causal relationship queries
    CausalRelationshipQuery,
}

/// Parameter types for visual reasoning queries
#[derive(Debug, Clone)]
pub enum QueryParameter {
    /// Text-based parameter
    Text(String),
    /// Numeric parameter
    Number(f32),
    /// Boolean parameter
    Boolean(bool),
    /// Image region specified as (x, y, width, height)
    ImageRegion((f32, f32, f32, f32)),
    /// Time range specified as (start, end)
    TimeRange((f32, f32)),
    /// List of object identifiers
    ObjectList(Vec<String>),
}

/// Context requirement for visual reasoning queries
#[derive(Debug, Clone)]
pub struct ContextRequirement {
    /// Type of context required
    pub requirement_type: String,
    /// Level of specificity needed (0.0-1.0)
    pub specificity: f32,
    /// Optional temporal scope for context
    pub temporal_scope: Option<(f32, f32)>,
}

/// Answer types for visual reasoning queries
#[derive(Debug, Clone)]
pub enum ReasoningAnswer {
    /// Text-based answer
    Text(String),
    /// Numeric answer
    Number(f32),
    /// Boolean answer
    Boolean(bool),
    /// List of detected objects
    ObjectList(Vec<String>),
    /// List of spatial locations
    LocationList(Vec<(f32, f32)>),
    /// Complex structured answer
    Complex(HashMap<String, String>),
}

/// Individual step in the reasoning process
#[derive(Debug, Clone)]
pub struct ReasoningStep {
    /// Unique identifier for this reasoning step
    pub step_id: usize,
    /// Type of reasoning operation performed
    pub step_type: String,
    /// Human-readable description of the step
    pub description: String,
    /// Input data used in this step
    pub input_data: Vec<String>,
    /// Output data generated by this step
    pub output_data: Vec<String>,
    /// Confidence in this reasoning step
    pub confidence: f32,
}

/// Evidence supporting a reasoning conclusion
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Type of evidence (visual, temporal, etc.)
    pub evidence_type: String,
    /// Description of the evidence
    pub description: String,
    /// Strength of support this evidence provides
    pub support_strength: f32,
    /// Visual locations that support this evidence
    pub visual_anchors: Vec<(f32, f32)>,
    /// Temporal points that support this evidence
    pub temporal_anchors: Vec<f32>,
}

/// Alternative hypothesis considered during reasoning
#[derive(Debug, Clone)]
pub struct AlternativeHypothesis {
    /// Description of the alternative hypothesis
    pub hypothesis: String,
    /// Probability or likelihood of this hypothesis
    pub probability: f32,
    /// Features that distinguish this from the main conclusion
    pub distinguishing_features: Vec<String>,
}

/// Quantification of uncertainty in reasoning results
#[derive(Debug, Clone)]
pub struct UncertaintyQuantification {
    /// Model uncertainty (knowledge limitations)
    pub epistemic_uncertainty: f32,
    /// Data uncertainty (inherent randomness)
    pub aleatoric_uncertainty: f32,
    /// Confidence interval for the answer
    pub confidence_interval: (f32, f32),
    /// Sensitivity to different input parameters
    pub sensitivity_analysis: HashMap<String, f32>,
}

// Additional supporting types
/// Model for causal relationships in visual scenes
#[derive(Debug, Clone)]
pub struct CausalModel {
    /// Name identifier for the causal model
    pub name: String,
    /// Variables involved in causal relationships
    pub variables: Vec<CausalVariable>,
    /// Causal relationships between variables
    pub relationships: Vec<CausalRelationship>,
    /// Overall confidence in the model
    pub confidence: f32,
}

/// Variable in a causal model
#[derive(Debug, Clone)]
pub struct CausalVariable {
    /// Name of the variable
    pub name: String,
    /// Type of the variable (continuous, discrete, etc.)
    pub variable_type: String,
    /// Possible values the variable can take
    pub possible_values: Vec<String>,
    /// How easily this variable can be observed
    pub observability: f32,
}

/// Relationship between cause and effect variables
#[derive(Debug, Clone)]
pub struct CausalRelationship {
    /// Variable that acts as the cause
    pub cause: String,
    /// Variable that is affected
    pub effect: String,
    /// Strength of the causal relationship
    pub strength: f32,
    /// Time delay between cause and effect
    pub delay: Option<f32>,
    /// Conditions under which this relationship holds
    pub conditions: Vec<String>,
}

/// Parameters for causal intervention analysis
#[derive(Debug, Clone)]
pub struct InterventionParams {
    /// Types of interventions to consider
    pub intervention_types: Vec<String>,
    /// Whether to model effect propagation through the graph
    pub effect_propagation: bool,
    /// Whether to include temporal aspects in modeling
    pub temporal_modeling: bool,
}

/// Parameters for counterfactual reasoning
#[derive(Debug, Clone)]
pub struct CounterfactualParams {
    /// Number of alternative scenarios to consider
    pub alternative_scenarios: usize,
    /// Threshold for considering scenarios plausible
    pub plausibility_threshold: f32,
    /// Temporal scope for counterfactual analysis
    pub temporal_scope: f32,
}

/// Types of questions that can be asked in visual reasoning
#[derive(Debug, Clone)]
pub enum QuestionType {
    /// Questions about objects in the scene
    Object,
    /// Questions about the overall scene
    Scene,
    /// Questions about activities or actions
    Activity,
    /// Questions about spatial relationships
    Spatial,
    /// Questions about temporal aspects
    Temporal,
    /// Questions about causal relationships
    Causal,
    /// Hypothetical "what if" questions
    Counterfactual,
    /// Questions comparing different elements
    Comparative,
}

/// Strategy for generating answers to visual reasoning queries
#[derive(Debug, Clone)]
pub struct AnswerStrategy {
    /// Name of the answer generation strategy
    pub strategy_name: String,
    /// Question types this strategy can handle
    pub applicable_types: Vec<QuestionType>,
    /// Whether this strategy provides confidence estimates
    pub confidence_estimation: bool,
}

/// Attention mechanism for focusing on relevant information
#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    /// Type of attention mechanism used
    pub mechanism_type: String,
    /// Whether spatial attention is enabled
    pub spatial_attention: bool,
    /// Whether temporal attention is enabled
    pub temporal_attention: bool,
    /// Whether cross-modal attention is enabled
    pub cross_modal_attention: bool,
}

/// Template for analogical reasoning between visual patterns
#[derive(Debug, Clone)]
pub struct AnalogyTemplate {
    /// Name of the analogy template
    pub template_name: String,
    /// Source pattern for the analogy
    pub source_pattern: VisualPattern,
    /// Target pattern for the analogy
    pub target_pattern: VisualPattern,
    /// Rules for mapping between source and target
    pub mapping_rules: Vec<MappingRule>,
}

/// Visual pattern representation for analogical reasoning
#[derive(Debug, Clone)]
pub struct VisualPattern {
    /// Type of visual pattern
    pub pattern_type: String,
    /// Feature representation of the pattern
    pub features: Array2<f32>,
    /// Spatial structure information
    pub spatial_structure: Array2<f32>,
    /// Temporal structure information
    pub temporal_structure: Array2<f32>,
}

/// Rule for mapping between elements in analogical reasoning
#[derive(Debug, Clone)]
pub struct MappingRule {
    /// Element in the source pattern
    pub source_element: String,
    /// Corresponding element in the target pattern
    pub target_element: String,
    /// Type of mapping relationship
    pub mapping_type: String,
    /// Confidence in this mapping
    pub confidence: f32,
}

/// Metric for computing similarity between visual patterns
#[derive(Debug, Clone)]
pub struct SimilarityMetric {
    /// Name of the similarity metric
    pub metric_name: String,
    /// Weights for different features
    pub feature_weights: Array1<f32>,
    /// Whether to normalize the metric
    pub normalization: bool,
    /// Distance function to use
    pub distance_function: String,
}

/// Parameters for transfer learning in visual reasoning
#[derive(Debug, Clone)]
pub struct TransferLearningParams {
    /// Rate of adaptation to new domains
    pub adaptation_rate: f32,
    /// Threshold for considering domains similar
    pub domain_similarity_threshold: f32,
    /// Whether to perform feature selection
    pub feature_selection: bool,
}

/// Detector for temporal events in visual sequences
#[derive(Debug, Clone)]
pub struct EventDetector {
    /// Type of event this detector recognizes
    pub event_type: String,
    /// Threshold for event detection
    pub detection_threshold: f32,
    /// Size of temporal window for detection
    pub temporal_window: usize,
    /// Feature extractors used for detection
    pub feature_extractors: Vec<String>,
}

/// Model for temporal relationships in visual reasoning
#[derive(Debug, Clone)]
pub struct TemporalModel {
    /// Type of temporal model
    pub model_type: String,
    /// Time horizon for predictions
    pub time_horizon: f32,
    /// Temporal granularity of the model
    pub granularity: f32,
    /// Whether to model causal relationships
    pub causality_modeling: bool,
}

/// Parameters for analyzing temporal sequences
#[derive(Debug, Clone)]
pub struct SequenceAnalysisParams {
    /// Maximum length of sequences to analyze
    pub max_sequence_length: usize,
    /// Whether to perform pattern recognition
    pub pattern_recognition: bool,
    /// Whether to detect anomalies in sequences
    pub anomaly_detection: bool,
}

/// Hierarchy of abstract concepts for visual reasoning
#[derive(Debug, Clone)]
pub struct ConceptHierarchy {
    /// Name of the concept hierarchy
    pub hierarchy_name: String,
    /// Root concepts at the top level
    pub root_concepts: Vec<String>,
    /// Relationships between concepts
    pub concept_relationships: HashMap<String, Vec<String>>,
    /// Number of abstraction levels
    pub abstraction_levels: usize,
}

/// Layer for feature abstraction in concept learning
#[derive(Debug, Clone)]
pub struct AbstractionLayer {
    /// Name of the abstraction layer
    pub layer_name: String,
    /// Number of input features
    pub input_features: usize,
    /// Number of output concepts
    pub output_concepts: usize,
    /// Learning algorithm used in this layer
    pub learning_algorithm: String,
}

/// Parameters for concept learning in visual reasoning
///
/// This structure configures how the system learns and emerges new concepts
/// from visual input data through adaptive mechanisms.
#[derive(Debug, Clone)]
pub struct ConceptLearningParams {
    /// Learning rate for concept adaptation and emergence
    pub learning_rate: f32,
    /// Threshold for determining when a new concept should emerge
    pub concept_emergence_threshold: f32,
    /// Whether to enable hierarchical concept learning
    pub hierarchical_learning: bool,
}

/// Different sensory modalities for multi-modal processing
///
/// Represents the various types of sensory input that can be processed
/// and fused in the visual reasoning system.
#[derive(Debug, Clone)]
pub enum Modality {
    /// Visual sensory input (images, video)
    Visual,
    /// Audio sensory input (sounds, speech)
    Audio,
    /// Textual input (natural language)
    Text,
    /// Tactile sensory input (touch, pressure)
    Tactile,
    /// Temporal sequence information
    Temporal,
    /// Spatial relationship information
    Spatial,
}

/// Strategy for fusing multiple sensory modalities
///
/// Defines how different sensory inputs should be combined and weighted
/// to create unified multi-modal representations.
#[derive(Debug, Clone)]
pub struct FusionStrategy {
    /// Name identifier for this fusion strategy
    pub strategy_name: String,
    /// Weights assigned to each modality in the fusion process
    pub modality_weights: HashMap<Modality, f32>,
    /// Level of fusion (early, intermediate, late)
    pub fusion_level: String,
    /// Whether to align temporal sequences across modalities
    pub temporal_alignment: bool,
}

/// Cross-modal attention mechanism for multi-modal processing
///
/// Implements attention mechanisms that allow one modality to attend to
/// and influence processing in another modality.
#[derive(Debug, Clone)]
pub struct CrossModalAttention {
    /// Type of attention mechanism (additive, multiplicative, etc.)
    pub attention_type: String,
    /// Source modality providing attention signal
    pub source_modality: Modality,
    /// Target modality receiving attention
    pub target_modality: Modality,
    /// Attention weight matrix
    pub attention_weights: Array2<f32>,
}

/// A visual fact extracted from reasoning about visual content
///
/// Represents a structured fact (subject-predicate-object triple) that has been
/// inferred or extracted from visual reasoning processes.
#[derive(Debug, Clone)]
pub struct VisualFact {
    /// Unique identifier for this fact
    pub fact_id: String,
    /// Subject of the fact (what the fact is about)
    pub subject: String,
    /// Predicate describing the relationship or property
    pub predicate: String,
    /// Object related to the subject by the predicate
    pub object: String,
    /// Confidence score for this fact (0.0 to 1.0)
    pub confidence: f32,
    /// Supporting evidence for this fact
    pub evidence: Vec<String>,
}

/// A logical reasoning rule for visual reasoning processes
///
/// Represents an if-then rule that can be applied during reasoning to derive
/// new conclusions from existing facts and conditions.
#[derive(Debug, Clone)]
pub struct ReasoningRule {
    /// Unique identifier for this reasoning rule
    pub rule_id: String,
    /// Conditions that must be met for the rule to apply
    pub conditions: Vec<String>,
    /// Conclusions that can be drawn when conditions are met
    pub conclusions: Vec<String>,
    /// Type of reasoning rule (deductive, inductive, abductive)
    pub rule_type: String,
    /// Reliability score for this rule (0.0 to 1.0)
    pub reliability: f32,
}

#[derive(Debug, Clone)]
pub struct ConceptOntology {
    pub concepts: HashMap<String, ConceptDefinition>,
    pub relationships: Vec<ConceptRelationship>,
    pub inheritance_hierarchy: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ConceptDefinition {
    pub concept_name: String,
    pub attributes: Vec<String>,
    pub visual_features: Array1<f32>,
    pub typical_contexts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConceptRelationship {
    pub source_concept: String,
    pub target_concept: String,
    pub relationship_type: String,
    pub strength: f32,
}

impl Default for VisualReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualReasoningEngine {
    /// Create a new advanced visual reasoning engine
    pub fn new() -> Self {
        Self {
            causal_inference: CausalInferenceModule::new(),
            vqa_system: VisualQuestionAnsweringSystem::new(),
            analogical_reasoning: AnalogicalReasoningEngine::new(),
            temporal_analyzer: TemporalEventAnalyzer::new(),
            concept_recognizer: AbstractConceptRecognizer::new(),
            multimodal_hub: MultiModalIntegrationHub::new(),
            knowledge_base: VisualKnowledgeBase::new(),
        }
    }

    /// Process a complex visual reasoning query
    pub fn process_query(
        &self,
        query: &VisualReasoningQuery,
        scene_analysis: &SceneAnalysisResult,
        context: Option<&[SceneAnalysisResult]>,
    ) -> Result<VisualReasoningResult> {
        // Initialize reasoning process
        let mut reasoning_steps = Vec::new();
        let mut evidence = Vec::new();

        // Step 1: Query understanding and decomposition
        let decomposed_query = self.decompose_query(query)?;
        reasoning_steps.push(ReasoningStep {
            step_id: 1,
            step_type: "query_decomposition".to_string(),
            description: "Breaking down complex query into sub-queries".to_string(),
            input_data: vec![query.question.clone()],
            output_data: vec![format!("{} sub-queries", decomposed_query.len())],
            confidence: 0.95,
        });

        // Step 2: Visual feature extraction and _analysis
        let visual_features = self.extract_reasoning_features(scene_analysis)?;
        reasoning_steps.push(ReasoningStep {
            step_id: 2,
            step_type: "feature_extraction".to_string(),
            description: "Extracting relevant visual features for reasoning".to_string(),
            input_data: vec!["scene_analysis".to_string()],
            output_data: vec![format!("{} feature dimensions", visual_features.len())],
            confidence: 0.90,
        });

        // Step 3: Apply reasoning based on query type
        let (answer, step_evidence, alternatives) = match query.query_type {
            QueryType::WhatIsHappening => {
                self.reason_what_is_happening(scene_analysis, &visual_features)?
            }
            QueryType::WhyIsHappening => {
                self.reason_why_is_happening(scene_analysis, &visual_features)?
            }
            QueryType::WhatWillHappenNext => {
                self.reason_what_will_happen_next(scene_analysis, context, &visual_features)?
            }
            QueryType::HowAreObjectsRelated => {
                self.reason_object_relationships(scene_analysis, &visual_features)?
            }
            QueryType::CausalRelationshipQuery => {
                self.reason_causal_relationships(scene_analysis, &visual_features)?
            }
            _ => (
                ReasoningAnswer::Text("Query type not fully implemented yet".to_string()),
                Vec::new(),
                Vec::new(),
            ),
        };

        evidence.extend(step_evidence);

        // Step 4: Confidence estimation and uncertainty quantification
        let confidence = self.estimate_overall_confidence(&reasoning_steps, &evidence)?;
        let uncertainty = self.quantify_uncertainty(&answer, &evidence)?;

        Ok(VisualReasoningResult {
            answer,
            reasoning_steps,
            confidence,
            evidence,
            alternatives,
            uncertainty,
        })
    }

    /// Process causal reasoning queries
    pub fn infer_causality(
        &self,
        scene_sequence: &[SceneAnalysisResult],
        causal_query: &str,
    ) -> Result<CausalInferenceResult> {
        // Extract temporal patterns
        let temporal_patterns = self.extract_temporal_patterns(scene_sequence)?;

        // Build causal graph
        let causal_graph = self
            .causal_inference
            .build_causal_graph(&temporal_patterns)?;

        // Perform causal inference
        let causal_effects = self
            .causal_inference
            .infer_effects(&causal_graph, causal_query)?;

        Ok(CausalInferenceResult {
            causal_graph,
            effects: causal_effects,
            confidence: 0.75,
        })
    }

    /// Perform analogical reasoning between scenes
    pub fn find_analogies(
        &self,
        source_scene: &SceneAnalysisResult,
        target_scenes: &[SceneAnalysisResult],
    ) -> Result<Vec<AnalogyResult>> {
        let mut analogies = Vec::new();

        for target_scene in target_scenes {
            let analogy = self
                .analogical_reasoning
                .find_analogy(source_scene, target_scene)?;
            if analogy.similarity_score > 0.6 {
                analogies.push(analogy);
            }
        }

        // Sort by similarity score
        analogies.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .expect("Operation failed")
        });

        Ok(analogies)
    }

    /// Recognize abstract concepts in visual scenes
    pub fn recognize_abstract_concepts(
        &self,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<Vec<AbstractConcept>> {
        let concepts = self.concept_recognizer.recognize_concepts(scene_analysis)?;
        Ok(concepts)
    }

    // Helper methods (placeholder implementations)
    fn decompose_query(&self, query: &VisualReasoningQuery) -> Result<Vec<SubQuery>> {
        // Placeholder implementation
        Ok(vec![SubQuery {
            sub_question: query.question.clone(),
            query_type: query.query_type.clone(),
            dependencies: Vec::new(),
        }])
    }

    fn extract_reasoning_features(
        &self,
        scene_analysis: &SceneAnalysisResult,
    ) -> Result<Array1<f32>> {
        // Extract multi-level features for reasoning
        let mut features = Vec::new();

        // Object-level features
        for object in &scene_analysis.objects {
            features.extend(object.features.iter().cloned());
        }

        // Relationship features
        for relationship in &scene_analysis.relationships {
            features.push(relationship.confidence);
            features.extend(relationship.parameters.values().cloned());
        }

        // Scene-level features
        features.push(scene_analysis.scene_confidence);

        Ok(Array1::from_vec(features))
    }

    fn reason_what_is_happening(
        &self,
        scene_analysis: &SceneAnalysisResult,
        _features: &Array1<f32>,
    ) -> Result<(ReasoningAnswer, Vec<Evidence>, Vec<AlternativeHypothesis>)> {
        // Analyze dominant activities and interactions
        let activities = self.identify_activities(scene_analysis)?;
        let description = format!("Detected activities: {}", activities.join(", "));

        let evidence = vec![Evidence {
            evidence_type: "object_detection".to_string(),
            description: format!("Found {} objects in scene", scene_analysis.objects.len()),
            support_strength: scene_analysis.scene_confidence,
            visual_anchors: scene_analysis
                .objects
                .iter()
                .map(|o| (o.bbox.0 + o.bbox.2 / 2.0, o.bbox.1 + o.bbox.3 / 2.0))
                .collect(),
            temporal_anchors: Vec::new(),
        }];

        Ok((ReasoningAnswer::Text(description), evidence, Vec::new()))
    }

    fn reason_why_is_happening(
        &self,
        scene_analysis: &SceneAnalysisResult,
        _features: &Array1<f32>,
    ) -> Result<(ReasoningAnswer, Vec<Evidence>, Vec<AlternativeHypothesis>)> {
        // Apply causal reasoning
        let causal_explanations = self.generate_causal_explanations(scene_analysis)?;

        Ok((
            ReasoningAnswer::Text(causal_explanations),
            Vec::new(),
            Vec::new(),
        ))
    }

    fn reason_what_will_happen_next(
        &self,
        scene_analysis: &SceneAnalysisResult,
        context: Option<&[SceneAnalysisResult]>,
        _features: &Array1<f32>,
    ) -> Result<(ReasoningAnswer, Vec<Evidence>, Vec<AlternativeHypothesis>)> {
        let prediction = if let Some(temporal_context) = context {
            self.predict_future_events(scene_analysis, temporal_context)?
        } else {
            "Insufficient temporal context for prediction".to_string()
        };

        Ok((ReasoningAnswer::Text(prediction), Vec::new(), Vec::new()))
    }

    fn reason_object_relationships(
        &self,
        scene_analysis: &SceneAnalysisResult,
        _features: &Array1<f32>,
    ) -> Result<(ReasoningAnswer, Vec<Evidence>, Vec<AlternativeHypothesis>)> {
        let relationships_desc = format!(
            "Found {} spatial relationships between objects",
            scene_analysis.relationships.len()
        );

        Ok((
            ReasoningAnswer::Text(relationships_desc),
            Vec::new(),
            Vec::new(),
        ))
    }

    fn reason_causal_relationships(
        &self,
        scene_analysis: &SceneAnalysisResult,
        _features: &Array1<f32>,
    ) -> Result<(ReasoningAnswer, Vec<Evidence>, Vec<AlternativeHypothesis>)> {
        let causal_analysis = self.analyze_causal_structure(scene_analysis)?;

        Ok((
            ReasoningAnswer::Text(causal_analysis),
            Vec::new(),
            Vec::new(),
        ))
    }

    /// Real confidence aggregation: the mean of the individual reasoning
    /// steps' `confidence` values and the evidence entries'
    /// `support_strength` values (equally weighted), falling back to a
    /// neutral `0.5` when there is nothing to aggregate. Replaces a
    /// previous unconditional `0.75` that ignored `steps`/`evidence`
    /// entirely.
    fn estimate_overall_confidence(
        &self,
        steps: &[ReasoningStep],
        evidence: &[Evidence],
    ) -> Result<f32> {
        let all_values: Vec<f32> = steps
            .iter()
            .map(|s| s.confidence)
            .chain(evidence.iter().map(|e| e.support_strength))
            .collect();
        if all_values.is_empty() {
            return Ok(0.5);
        }
        let mean = all_values.iter().sum::<f32>() / all_values.len() as f32;
        Ok(mean.clamp(0.0, 1.0))
    }

    /// `confidence_interval` and `sensitivity_analysis` are computed for
    /// real from `evidence`'s actual `support_strength` values (mean +/- one
    /// standard deviation, and mean strength grouped by `evidence_type`,
    /// respectively) rather than fixed constants. `epistemic_uncertainty`/
    /// `aleatoric_uncertainty` -- a principled model-vs-data uncertainty
    /// *decomposition* -- would need a trained/probabilistic model this
    /// crate does not have, so those two fields remain a documented
    /// placeholder (see the module-level doc comment) rather than an
    /// invented split.
    fn quantify_uncertainty(
        &self,
        _answer: &ReasoningAnswer,
        evidence: &[Evidence],
    ) -> Result<UncertaintyQuantification> {
        let confidence_interval = if evidence.is_empty() {
            (0.5, 0.5)
        } else {
            let strengths: Vec<f32> = evidence.iter().map(|e| e.support_strength).collect();
            let mean = strengths.iter().sum::<f32>() / strengths.len() as f32;
            let variance =
                strengths.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / strengths.len() as f32;
            let std_dev = variance.sqrt();
            (
                (mean - std_dev).clamp(0.0, 1.0),
                (mean + std_dev).clamp(0.0, 1.0),
            )
        };

        let mut sensitivity_sums: HashMap<String, (f32, usize)> = HashMap::new();
        for e in evidence {
            let entry = sensitivity_sums
                .entry(e.evidence_type.clone())
                .or_insert((0.0, 0));
            entry.0 += e.support_strength;
            entry.1 += 1;
        }
        let sensitivity_analysis = sensitivity_sums
            .into_iter()
            .map(|(evidence_type, (sum, count))| (evidence_type, sum / count as f32))
            .collect();

        Ok(UncertaintyQuantification {
            epistemic_uncertainty: 0.2,
            aleatoric_uncertainty: 0.1,
            confidence_interval,
            sensitivity_analysis,
        })
    }

    fn extract_temporal_patterns(
        &self,
        sequence: &[SceneAnalysisResult],
    ) -> Result<TemporalPatterns> {
        Ok(TemporalPatterns {
            patterns: Vec::new(),
            temporal_graph: TemporalGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
        })
    }

    fn identify_activities(&self, sceneanalysis: &SceneAnalysisResult) -> Result<Vec<String>> {
        let mut activities = Vec::new();

        // Analyze object combinations and spatial relationships
        for object in &sceneanalysis.objects {
            match object.class.as_str() {
                "person" => activities.push("human_activity".to_string()),
                "car" => activities.push("transportation".to_string()),
                "chair" => activities.push("sitting_area".to_string()),
                _ => {}
            }
        }

        if activities.is_empty() {
            activities.push("static_scene".to_string());
        }

        Ok(activities)
    }

    /// Surface the *real* rule-based conclusions
    /// [`SceneAnalysisResult::reasoning_results`] already computed by
    /// [`crate::scene_understanding`]'s [`ContextualReasoningEngine`], rather
    /// than a fixed generic sentence. This is not novel causal inference
    /// (that would need a trained model this crate doesn't have); it
    /// honestly reports what the classical reasoning rules already
    /// concluded, or says plainly that no rule fired.
    ///
    /// [`ContextualReasoningEngine`]: crate::scene_understanding::ContextualReasoningEngine
    fn generate_causal_explanations(&self, scene_analysis: &SceneAnalysisResult) -> Result<String> {
        if scene_analysis.reasoning_results.is_empty() {
            return Ok(format!(
                "No reasoning rule matched this scene ({} objects, {} relationships); \
                 no explanation available.",
                scene_analysis.objects.len(),
                scene_analysis.relationships.len()
            ));
        }

        let explanations: Vec<String> = scene_analysis
            .reasoning_results
            .iter()
            .map(|r| format!("{} (confidence {:.2})", r.conclusion, r.confidence))
            .collect();
        Ok(explanations.join("; "))
    }

    /// Compare the current scene's object count against the trailing
    /// temporal `context` to report a real (if coarse) stability judgement,
    /// rather than an unconditional "likely to remain stable" regardless of
    /// input. This is a heuristic trend read on object *count*, not genuine
    /// event prediction.
    fn predict_future_events(
        &self,
        scene: &SceneAnalysisResult,
        context: &[SceneAnalysisResult],
    ) -> Result<String> {
        if context.is_empty() {
            return Ok("Insufficient temporal context for prediction".to_string());
        }

        let mean_context_count =
            context.iter().map(|s| s.objects.len() as f32).sum::<f32>() / context.len() as f32;
        let current_count = scene.objects.len() as f32;
        let delta = current_count - mean_context_count;

        let trend = if delta.abs() < 0.5 {
            "stable (object count roughly unchanged)"
        } else if delta > 0.0 {
            "increasingly active (object count rising)"
        } else {
            "quieting down (object count falling)"
        };

        Ok(format!(
            "Based on {} prior frame(s) averaging {:.1} objects vs. {} now, \
             the scene appears {trend}.",
            context.len(),
            mean_context_count,
            scene.objects.len()
        ))
    }

    /// Summarize the real spatial relationships already detected by
    /// [`crate::scene_understanding`] as candidate causal structure (spatial
    /// co-location is evidence for, not proof of, a causal link) rather than
    /// an unconditional "no relationships" message.
    fn analyze_causal_structure(&self, scene_analysis: &SceneAnalysisResult) -> Result<String> {
        if scene_analysis.relationships.is_empty() {
            return Ok(
                "No spatial relationships detected in current scene; no candidate \
                causal structure to report."
                    .to_string(),
            );
        }

        Ok(format!(
            "{} spatial relationship(s) detected between objects, offering candidate (not \
             confirmed) causal structure; mean relationship confidence {:.2}.",
            scene_analysis.relationships.len(),
            scene_analysis
                .relationships
                .iter()
                .map(|r| r.confidence)
                .sum::<f32>()
                / scene_analysis.relationships.len() as f32
        ))
    }
}

// Placeholder structures for compilation
#[derive(Debug, Clone)]
pub struct SubQuery {
    pub sub_question: String,
    pub query_type: QueryType,
    pub dependencies: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct CausalInferenceResult {
    pub causal_graph: CausalGraph,
    pub effects: Vec<CausalEffect>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub nodes: Vec<CausalNode>,
    pub edges: Vec<CausalEdge>,
}

#[derive(Debug, Clone)]
pub struct CausalNode {
    pub node_id: String,
    pub node_type: String,
    pub properties: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct CausalEdge {
    pub source: String,
    pub target: String,
    pub strength: f32,
    pub delay: f32,
}

#[derive(Debug, Clone)]
pub struct CausalEffect {
    pub effect_type: String,
    pub magnitude: f32,
    pub probability: f32,
}

#[derive(Debug, Clone)]
pub struct AnalogyResult {
    pub similarity_score: f32,
    pub matching_patterns: Vec<PatternMatch>,
    pub explanation: String,
}

#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub source_element: String,
    pub target_element: String,
    pub similarity: f32,
}

#[derive(Debug, Clone)]
pub struct AbstractConcept {
    pub concept_name: String,
    pub confidence: f32,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TemporalPatterns {
    pub patterns: Vec<TemporalPattern>,
    pub temporal_graph: TemporalGraph,
}

#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_type: String,
    pub frequency: f32,
    pub duration: f32,
}

#[derive(Debug, Clone)]
pub struct TemporalGraph {
    pub nodes: Vec<TemporalNode>,
    pub edges: Vec<TemporalEdge>,
}

#[derive(Debug, Clone)]
pub struct TemporalNode {
    pub timestamp: f32,
    pub event_type: String,
    pub properties: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct TemporalEdge {
    pub source_time: f32,
    pub target_time: f32,
    pub relationship_type: String,
}

// Implementation stubs for associated types
impl CausalInferenceModule {
    fn new() -> Self {
        Self {
            causal_models: Vec::new(),
            intervention_params: InterventionParams {
                intervention_types: Vec::new(),
                effect_propagation: true,
                temporal_modeling: true,
            },
            counterfactual_params: CounterfactualParams {
                alternative_scenarios: 5,
                plausibility_threshold: 0.3,
                temporal_scope: 10.0,
            },
        }
    }

    fn build_causal_graph(&self, patterns: &TemporalPatterns) -> Result<CausalGraph> {
        Ok(CausalGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        })
    }

    fn infer_effects(&self, graph: &CausalGraph, query: &str) -> Result<Vec<CausalEffect>> {
        Ok(Vec::new())
    }
}

impl VisualQuestionAnsweringSystem {
    fn new() -> Self {
        Self {
            question_types: vec![QuestionType::Object, QuestionType::Scene],
            answer_strategies: Vec::new(),
            attention_mechanisms: Vec::new(),
        }
    }
}

impl AnalogicalReasoningEngine {
    fn new() -> Self {
        Self {
            analogy_templates: Vec::new(),
            similarity_metrics: Vec::new(),
            transfer_params: TransferLearningParams {
                adaptation_rate: 0.1,
                domain_similarity_threshold: 0.5,
                feature_selection: true,
            },
        }
    }

    /// Real (classical, non-learned) structural-similarity analogy: how much
    /// two scenes' object-class composition, object count, and relationship
    /// count resemble each other. This is not learned analogical mapping
    /// (genuinely out of scope without a trained model, per the module doc),
    /// but a real, deterministic feature comparison rather than a fixed
    /// `0.7` regardless of the two scenes' actual content.
    fn find_analogy(
        &self,
        source: &SceneAnalysisResult,
        target: &SceneAnalysisResult,
    ) -> Result<AnalogyResult> {
        let source_classes: std::collections::HashSet<&str> =
            source.objects.iter().map(|o| o.class.as_str()).collect();
        let target_classes: std::collections::HashSet<&str> =
            target.objects.iter().map(|o| o.class.as_str()).collect();

        let intersection = source_classes.intersection(&target_classes).count();
        let union = source_classes.union(&target_classes).count().max(1);
        let class_similarity = intersection as f32 / union as f32;

        let ratio_similarity = |a: usize, b: usize| -> f32 {
            let (a, b) = (a as f32, b as f32);
            if a.max(b) > 0.0 {
                1.0 - (a - b).abs() / a.max(b)
            } else {
                1.0
            }
        };
        let count_similarity = ratio_similarity(source.objects.len(), target.objects.len());
        let relationship_similarity =
            ratio_similarity(source.relationships.len(), target.relationships.len());

        let similarity_score =
            (class_similarity + count_similarity + relationship_similarity) / 3.0;

        let mut matching_patterns: Vec<PatternMatch> = source_classes
            .intersection(&target_classes)
            .map(|&class| PatternMatch {
                source_element: class.to_string(),
                target_element: class.to_string(),
                similarity: 1.0,
            })
            .collect();
        matching_patterns.sort_by(|a, b| a.source_element.cmp(&b.source_element));

        let explanation = if matching_patterns.is_empty() {
            format!(
                "No shared object classes between scenes ({} vs {} objects); \
                 similarity score {similarity_score:.2} reflects only count/relationship overlap.",
                source.objects.len(),
                target.objects.len()
            )
        } else {
            let shared: Vec<&str> = matching_patterns
                .iter()
                .map(|m| m.source_element.as_str())
                .collect();
            format!(
                "Shared object classes: {}; similarity score {similarity_score:.2} combines \
                 class, count, and relationship overlap.",
                shared.join(", ")
            )
        };

        Ok(AnalogyResult {
            similarity_score,
            matching_patterns,
            explanation,
        })
    }
}

impl TemporalEventAnalyzer {
    fn new() -> Self {
        Self {
            event_detectors: Vec::new(),
            temporal_models: Vec::new(),
            sequence_params: SequenceAnalysisParams {
                max_sequence_length: 100,
                pattern_recognition: true,
                anomaly_detection: true,
            },
        }
    }
}

impl AbstractConceptRecognizer {
    fn new() -> Self {
        Self {
            concept_hierarchies: Vec::new(),
            abstraction_layers: Vec::new(),
            learning_params: ConceptLearningParams {
                learning_rate: 0.01,
                concept_emergence_threshold: 0.8,
                hierarchical_learning: true,
            },
        }
    }

    fn recognize_concepts(&self, scene: &SceneAnalysisResult) -> Result<Vec<AbstractConcept>> {
        Ok(Vec::new())
    }
}

impl MultiModalIntegrationHub {
    fn new() -> Self {
        Self {
            modalities: vec![Modality::Visual],
            fusion_strategies: Vec::new(),
            cross_attention: Vec::new(),
        }
    }
}

impl VisualKnowledgeBase {
    fn new() -> Self {
        Self {
            facts: HashMap::new(),
            rules: Vec::new(),
            ontology: ConceptOntology {
                concepts: HashMap::new(),
                relationships: Vec::new(),
                inheritance_hierarchy: HashMap::new(),
            },
        }
    }
}

/// High-level function for complex visual reasoning
#[allow(dead_code)]
pub fn perform_advanced_visual_reasoning(
    scene: &SceneAnalysisResult,
    question: &str,
    context: Option<&[SceneAnalysisResult]>,
) -> Result<VisualReasoningResult> {
    let engine = VisualReasoningEngine::new();

    let query = VisualReasoningQuery {
        query_type: QueryType::WhatIsHappening, // Default, could be inferred from question
        question: question.to_string(),
        parameters: HashMap::new(),
        context_requirements: Vec::new(),
    };

    engine.process_query(&query, scene, context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene_understanding::{
        DetectedObject, ReasoningResult, SceneGraph, SpatialRelation, SpatialRelationType,
    };

    fn object(class: &str, bbox: (f32, f32, f32, f32)) -> DetectedObject {
        DetectedObject {
            class: class.to_string(),
            bbox,
            confidence: 0.9,
            features: Array2::zeros((1, 4)),
            mask: None,
            attributes: HashMap::new(),
        }
    }

    fn relation(source_id: usize, target_id: usize, confidence: f32) -> SpatialRelation {
        SpatialRelation {
            source_id,
            target_id,
            relation_type: SpatialRelationType::NextTo,
            confidence,
            parameters: HashMap::new(),
        }
    }

    fn scene(
        objects: Vec<DetectedObject>,
        relationships: Vec<SpatialRelation>,
        reasoning_results: Vec<ReasoningResult>,
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
            reasoning_results,
        }
    }

    #[test]
    fn test_generate_causal_explanations_uses_real_reasoning_results() {
        let engine = VisualReasoningEngine::new();

        let empty = scene(Vec::new(), Vec::new(), Vec::new());
        let empty_explanation = engine
            .generate_causal_explanations(&empty)
            .expect("generate_causal_explanations failed");
        assert!(empty_explanation.contains("No reasoning rule matched"));

        let with_results = scene(
            Vec::new(),
            Vec::new(),
            vec![ReasoningResult {
                rule_name: "test_rule".to_string(),
                conclusion: "objects are clustered".to_string(),
                confidence: 0.42,
                evidence: Vec::new(),
            }],
        );
        let real_explanation = engine
            .generate_causal_explanations(&with_results)
            .expect("generate_causal_explanations failed");
        assert!(real_explanation.contains("objects are clustered"));
        assert!(real_explanation.contains("0.42"));
        assert_ne!(real_explanation, empty_explanation);
    }

    #[test]
    fn test_predict_future_events_reads_real_trend_not_hardcoded() {
        let engine = VisualReasoningEngine::new();

        let no_context = scene(
            vec![object("person", (0.0, 0.0, 1.0, 1.0))],
            Vec::new(),
            Vec::new(),
        );
        let no_context_result = engine
            .predict_future_events(&no_context, &[])
            .expect("predict_future_events failed");
        assert_eq!(
            no_context_result,
            "Insufficient temporal context for prediction"
        );

        let quiet_history = vec![
            scene(Vec::new(), Vec::new(), Vec::new()),
            scene(Vec::new(), Vec::new(), Vec::new()),
        ];
        let busy_now = scene(
            vec![
                object("person", (0.0, 0.0, 1.0, 1.0)),
                object("person", (2.0, 0.0, 1.0, 1.0)),
                object("car", (4.0, 0.0, 1.0, 1.0)),
            ],
            Vec::new(),
            Vec::new(),
        );
        let trend_result = engine
            .predict_future_events(&busy_now, &quiet_history)
            .expect("predict_future_events failed");
        assert!(
            trend_result.contains("increasingly active"),
            "expected an activity increase to be detected, got: {trend_result}"
        );
        assert_ne!(
            trend_result,
            "Based on temporal patterns, the _scene is likely to remain stable"
        );
    }

    #[test]
    fn test_analyze_causal_structure_reports_real_relationship_count() {
        let engine = VisualReasoningEngine::new();

        let none = scene(Vec::new(), Vec::new(), Vec::new());
        let none_result = engine
            .analyze_causal_structure(&none)
            .expect("analyze_causal_structure failed");
        assert!(none_result.contains("No spatial relationships"));

        let with_rels = scene(
            vec![
                object("object", (0.0, 0.0, 1.0, 1.0)),
                object("object", (1.0, 1.0, 1.0, 1.0)),
            ],
            vec![relation(0, 1, 0.6), relation(1, 0, 0.8)],
            Vec::new(),
        );
        let with_rels_result = engine
            .analyze_causal_structure(&with_rels)
            .expect("analyze_causal_structure failed");
        assert!(
            with_rels_result.contains('2'),
            "should report the real count of 2 relationships"
        );
        assert!(
            with_rels_result.contains("0.70"),
            "mean confidence of 0.6 and 0.8 is 0.70"
        );
    }

    #[test]
    fn test_find_analogy_computes_real_similarity_not_hardcoded() {
        let engine = VisualReasoningEngine::new();

        let scene_a = scene(
            vec![
                object("person", (0.0, 0.0, 1.0, 1.0)),
                object("car", (1.0, 0.0, 1.0, 1.0)),
            ],
            vec![relation(0, 1, 0.5)],
            Vec::new(),
        );
        let identical = scene(
            vec![
                object("person", (0.0, 0.0, 1.0, 1.0)),
                object("car", (1.0, 0.0, 1.0, 1.0)),
            ],
            vec![relation(0, 1, 0.5)],
            Vec::new(),
        );
        let disjoint = scene(
            vec![
                object("chair", (0.0, 0.0, 1.0, 1.0)),
                object("table", (1.0, 0.0, 1.0, 1.0)),
                object("lamp", (2.0, 0.0, 1.0, 1.0)),
            ],
            Vec::new(),
            Vec::new(),
        );

        let identical_analogy = engine
            .analogical_reasoning
            .find_analogy(&scene_a, &identical)
            .expect("find_analogy failed");
        let disjoint_analogy = engine
            .analogical_reasoning
            .find_analogy(&scene_a, &disjoint)
            .expect("find_analogy failed");

        assert!(
            (identical_analogy.similarity_score - 1.0).abs() < 1e-6,
            "identical scenes should score ~1.0, got {}",
            identical_analogy.similarity_score
        );
        assert!(
            disjoint_analogy.similarity_score < identical_analogy.similarity_score,
            "a scene with no shared classes must score lower"
        );
        assert_ne!(disjoint_analogy.similarity_score, 0.7);
        assert_eq!(identical_analogy.matching_patterns.len(), 2);
    }

    #[test]
    fn test_quantify_uncertainty_uses_real_evidence_spread() {
        let engine = VisualReasoningEngine::new();
        let answer = ReasoningAnswer::Text("test".to_string());

        let empty = engine
            .quantify_uncertainty(&answer, &[])
            .expect("quantify_uncertainty failed");
        assert_eq!(empty.confidence_interval, (0.5, 0.5));

        let agreeing = engine
            .quantify_uncertainty(&answer, &[evidence(0.8), evidence(0.8)])
            .expect("quantify_uncertainty failed");
        assert!(
            (agreeing.confidence_interval.1 - agreeing.confidence_interval.0).abs() < 1e-6,
            "identical evidence should yield a zero-width interval, got {:?}",
            agreeing.confidence_interval
        );

        let disagreeing = engine
            .quantify_uncertainty(&answer, &[evidence(0.1), evidence(0.9)])
            .expect("quantify_uncertainty failed");
        assert!(
            disagreeing.confidence_interval.1 - disagreeing.confidence_interval.0
                > agreeing.confidence_interval.1 - agreeing.confidence_interval.0,
            "disagreeing evidence must widen the interval"
        );
    }

    fn step(confidence: f32) -> ReasoningStep {
        ReasoningStep {
            step_id: 0,
            step_type: "test".to_string(),
            description: "test step".to_string(),
            input_data: Vec::new(),
            output_data: Vec::new(),
            confidence,
        }
    }

    fn evidence(support_strength: f32) -> Evidence {
        Evidence {
            evidence_type: "test".to_string(),
            description: "test evidence".to_string(),
            support_strength,
            visual_anchors: Vec::new(),
            temporal_anchors: Vec::new(),
        }
    }

    #[test]
    fn test_estimate_overall_confidence_responds_to_inputs() {
        // Regression guard: the original implementation returned an
        // unconditional `0.75` regardless of `steps`/`evidence`.
        let engine = VisualReasoningEngine::new();

        let empty_confidence = engine
            .estimate_overall_confidence(&[], &[])
            .expect("estimate_overall_confidence failed");
        assert_eq!(empty_confidence, 0.5);

        let high_confidence = engine
            .estimate_overall_confidence(&[step(0.95), step(0.9)], &[evidence(0.85)])
            .expect("estimate_overall_confidence failed");
        let low_confidence = engine
            .estimate_overall_confidence(&[step(0.1), step(0.05)], &[evidence(0.15)])
            .expect("estimate_overall_confidence failed");

        assert!(high_confidence > 0.8);
        assert!(low_confidence < 0.2);
        assert!(high_confidence > low_confidence);
    }
}
