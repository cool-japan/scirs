//! Memory Systems for Hyperdimensional Computing
//!
//! This module implements various memory systems for HDC including basic storage,
//! continual learning with interference prevention, online learning systems,
//! and performance tracking for adaptive learning algorithms.

use std::collections::{HashMap, VecDeque};

use crate::error::{NdimageError, NdimageResult};
use crate::hyperdimensional_computing::types::{
    AdaptationParameters, ConsolidationResult, Experience, HDCConfig, Hypervector,
    OnlineLearningResult, PerformanceMetrics, PredictionResult, UpdateResult,
};

/// Basic HDC Memory for storing and retrieving patterns
#[derive(Debug, Clone)]
pub struct HDCMemory {
    /// Stored hypervector patterns with labels
    pub patterns: HashMap<String, Hypervector>,
    /// Item memory for atomic concepts
    pub item_memory: HashMap<String, Hypervector>,
    /// Configuration parameters
    pub config: HDCConfig,
}

impl HDCMemory {
    /// Create a new HDC memory
    pub fn new(config: HDCConfig) -> Self {
        Self {
            patterns: HashMap::new(),
            item_memory: HashMap::new(),
            config,
        }
    }

    /// Store a pattern with its label
    ///
    /// # Arguments
    ///
    /// * `label` - Label for the pattern
    /// * `pattern` - Hypervector pattern to store
    pub fn store(&mut self, label: String, pattern: Hypervector) {
        self.patterns.insert(label, pattern);
    }

    /// Retrieve the best matching pattern
    ///
    /// # Arguments
    ///
    /// * `query` - Query hypervector
    ///
    /// # Returns
    ///
    /// Option containing (matched_label, confidence) if a match is found
    pub fn retrieve(&self, query: &Hypervector) -> Option<(String, f64)> {
        let mut best_match = None;
        let mut best_similarity = 0.0;

        for (label, pattern) in &self.patterns {
            let similarity = query.similarity(pattern);
            if similarity > best_similarity && similarity >= self.config.similarity_threshold {
                best_similarity = similarity;
                best_match = Some((label.clone(), similarity));
            }
        }

        best_match
    }

    /// Get all stored patterns
    pub fn get_patterns(&self) -> &HashMap<String, Hypervector> {
        &self.patterns
    }

    /// Remove a pattern
    ///
    /// # Arguments
    ///
    /// * `label` - Label of pattern to remove
    ///
    /// # Returns
    ///
    /// The removed pattern if it existed
    pub fn remove(&mut self, label: &str) -> Option<Hypervector> {
        self.patterns.remove(label)
    }

    /// Clear all stored patterns
    pub fn clear(&mut self) {
        self.patterns.clear();
    }

    /// Get number of stored patterns
    pub fn size(&self) -> usize {
        self.patterns.len()
    }

    /// Update an existing pattern or store a new one
    ///
    /// # Arguments
    ///
    /// * `label` - Pattern label
    /// * `new_pattern` - New pattern to store
    /// * `learning_rate` - Rate for updating existing patterns
    pub fn update_pattern(
        &mut self,
        label: String,
        new_pattern: Hypervector,
        learning_rate: f64,
    ) -> NdimageResult<()> {
        if let Some(existing_pattern) = self.patterns.get(&label) {
            // Bundle with existing pattern using learning rate
            let weighted_new = new_pattern.scale(learning_rate);
            let weighted_existing = existing_pattern.scale(1.0 - learning_rate);
            let updated = weighted_existing.bundle(&weighted_new)?;
            self.patterns.insert(label, updated);
        } else {
            self.patterns.insert(label, new_pattern);
        }
        Ok(())
    }

    /// Store item in item memory
    pub fn store_item(&mut self, name: String, item: Hypervector) {
        self.item_memory.insert(name, item);
    }

    /// Get item from item memory
    pub fn get_item(&self, name: &str) -> Option<&Hypervector> {
        self.item_memory.get(name)
    }
}

/// Continual Learning Memory with interference prevention
#[derive(Debug, Clone)]
pub struct ContinualLearningMemory {
    /// Main associative memory
    pub associative_memory: HashMap<String, Hypervector>,
    /// Episodic buffer for recent experiences
    pub episodic_buffer: VecDeque<Experience>,
    /// Long-term consolidated memories
    pub consolidated_memory: HashMap<String, Hypervector>,
    /// Configuration parameters
    pub config: HDCConfig,
    /// Maximum size of episodic buffer
    pub buffer_capacity: usize,
    /// Interference threshold
    pub interference_threshold: f64,
}

impl ContinualLearningMemory {
    /// Create a new continual learning memory
    pub fn new(config: &HDCConfig) -> Self {
        Self {
            associative_memory: HashMap::new(),
            episodic_buffer: VecDeque::new(),
            consolidated_memory: HashMap::new(),
            config: config.clone(),
            buffer_capacity: 1000,
            interference_threshold: 0.7,
        }
    }

    /// Add a new experience with consolidation
    ///
    /// # Arguments
    ///
    /// * `experience` - New experience to add
    /// * `consolidation` - Consolidation result information
    pub fn add_experience(
        &mut self,
        experience: Experience,
        _consolidation: &ConsolidationResult,
    ) -> NdimageResult<()> {
        // Check for interference
        let interference = self.calculate_interference(&experience.encoding);

        if interference > self.interference_threshold {
            self.perform_replay_consolidation(&experience)?;
        }

        // Add to episodic buffer
        self.episodic_buffer.push_back(experience.clone());

        // Maintain buffer capacity
        if self.episodic_buffer.len() > self.buffer_capacity {
            self.episodic_buffer.pop_front();
        }

        // Update associative memory
        self.associative_memory
            .insert(experience.label.clone(), experience.encoding);

        Ok(())
    }

    /// Calculate interference with existing memories
    ///
    /// # Arguments
    ///
    /// * `new_encoding` - New encoding to check for interference
    ///
    /// # Returns
    ///
    /// Interference score between 0.0 and 1.0
    pub fn calculate_interference(&self, new_encoding: &Hypervector) -> f64 {
        let mut max_interference = 0.0;

        for (_, existing_encoding) in &self.associative_memory {
            let similarity = new_encoding.similarity(existing_encoding);
            if similarity > max_interference {
                max_interference = similarity;
            }
        }

        max_interference
    }

    /// Perform replay-based consolidation
    fn perform_replay_consolidation(&mut self, new_experience: &Experience) -> NdimageResult<()> {
        // Simplified replay consolidation
        // In practice, this would implement more sophisticated algorithms

        // Find most similar existing experience
        let mut most_similar_label = None;
        let mut max_similarity = 0.0;

        for experience in &self.episodic_buffer {
            let similarity = new_experience.encoding.similarity(&experience.encoding);
            if similarity > max_similarity {
                max_similarity = similarity;
                most_similar_label = Some(experience.label.clone());
            }
        }

        // If highly similar experience exists, consolidate
        if let Some(label) = most_similar_label {
            if max_similarity > self.interference_threshold {
                if let Some(existing) = self.associative_memory.get(&label) {
                    let consolidated = existing.bundle(&new_experience.encoding)?;
                    self.consolidated_memory.insert(label, consolidated);
                }
            }
        }

        Ok(())
    }

    /// Retrieve from memory with consolidation awareness
    pub fn retrieve(&self, query: &Hypervector) -> Option<(String, f64)> {
        let mut best_match = None;
        let mut best_similarity = 0.0;

        // Check associative memory first
        for (label, pattern) in &self.associative_memory {
            let similarity = query.similarity(pattern);
            if similarity > best_similarity && similarity >= self.config.similarity_threshold {
                best_similarity = similarity;
                best_match = Some((label.clone(), similarity));
            }
        }

        // Check consolidated memory
        for (label, pattern) in &self.consolidated_memory {
            let similarity = query.similarity(pattern);
            if similarity > best_similarity && similarity >= self.config.similarity_threshold {
                best_similarity = similarity;
                best_match = Some((label.clone(), similarity));
            }
        }

        best_match
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> (usize, usize, usize) {
        (
            self.associative_memory.len(),
            self.episodic_buffer.len(),
            self.consolidated_memory.len(),
        )
    }

    /// Get current timestamp
    pub fn get_current_time(&self) -> usize {
        // Simple timestamp based on number of experiences
        self.episodic_buffer.len()
    }

    /// Update meta-learning parameters (placeholder)
    pub fn update_meta_learning_parameters(
        &mut self,
        _stats: &crate::hyperdimensional_computing::reasoning::ContinualLearningStats,
    ) {
        // Placeholder implementation
    }

    /// Get meta-learning score (placeholder)
    pub fn get_meta_learning_score(&self) -> f64 {
        // Placeholder implementation
        0.7
    }
}

/// Performance tracker for adaptive learning
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// History of accuracy measurements
    pub accuracyhistory: VecDeque<f64>,
    /// History of learning speed measurements
    pub learning_speedhistory: VecDeque<f64>,
    /// Number of updates performed
    pub update_count: usize,
    /// Maximum history length
    pub max_history_length: usize,
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new() -> Self {
        Self {
            accuracyhistory: VecDeque::new(),
            learning_speedhistory: VecDeque::new(),
            update_count: 0,
            max_history_length: 100,
        }
    }

    /// Record a performance update
    ///
    /// # Arguments
    ///
    /// * `accuracy` - Current accuracy measurement
    /// * `learning_speed` - Current learning speed measurement
    pub fn record_update(&mut self, accuracy: f64, learning_speed: f64) {
        self.accuracyhistory.push_back(accuracy);
        self.learning_speedhistory.push_back(learning_speed);
        self.update_count += 1;

        // Maintain maximum history length
        if self.accuracyhistory.len() > self.max_history_length {
            self.accuracyhistory.pop_front();
        }
        if self.learning_speedhistory.len() > self.max_history_length {
            self.learning_speedhistory.pop_front();
        }
    }

    /// Get current accuracy estimate
    pub fn get_accuracy(&self) -> f64 {
        if self.accuracyhistory.is_empty() {
            0.0
        } else {
            self.accuracyhistory.iter().sum::<f64>() / self.accuracyhistory.len() as f64
        }
    }

    /// Get current learning speed estimate
    pub fn get_learning_speed(&self) -> f64 {
        if self.learning_speedhistory.is_empty() {
            0.0
        } else {
            self.learning_speedhistory.iter().sum::<f64>() / self.learning_speedhistory.len() as f64
        }
    }

    /// Get memory efficiency estimate
    pub fn get_memory_efficiency(&self) -> f64 {
        // Simplified memory efficiency calculation
        1.0 / (1.0 + self.update_count as f64 / 1000.0)
    }

    /// Get recent performance change
    pub fn get_recent_performance_change(&self) -> f64 {
        if self.accuracyhistory.len() < 10 {
            return 0.0;
        }

        let recent: f64 = self.accuracyhistory.iter().rev().take(5).sum::<f64>() / 5.0;
        let older: f64 = self
            .accuracyhistory
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .sum::<f64>()
            / 5.0;
        recent - older
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.accuracyhistory.clear();
        self.learning_speedhistory.clear();
        self.update_count = 0;
    }
}

/// Online Learning System with adaptive capabilities
#[derive(Debug, Clone)]
pub struct OnlineLearningSystem {
    /// Core memory system
    pub memory: HDCMemory,
    /// Continual learning memory
    pub continual_memory: ContinualLearningMemory,
    /// Performance tracker
    pub performance_tracker: PerformanceTracker,
    /// Adaptation parameters
    pub adaptation_params: AdaptationParameters,
    /// Current learning state
    pub learning_state: LearningState,
}

/// Learning state enumeration
#[derive(Debug, Clone)]
pub enum LearningState {
    /// Normal learning mode
    Normal,
    /// Rapid adaptation mode
    RapidAdaptation,
    /// Conservative mode (preventing interference)
    Conservative,
}

impl OnlineLearningSystem {
    /// Create a new online learning system
    pub fn new(config: &HDCConfig) -> Self {
        Self {
            memory: HDCMemory::new(config.clone()),
            continual_memory: ContinualLearningMemory::new(config),
            performance_tracker: PerformanceTracker::new(),
            adaptation_params: AdaptationParameters::default(),
            learning_state: LearningState::Normal,
        }
    }

    /// Make a prediction
    ///
    /// # Arguments
    ///
    /// * `input` - Input hypervector to classify
    ///
    /// # Returns
    ///
    /// Prediction result with confidence and alternatives
    pub fn predict(&self, input: &Hypervector) -> NdimageResult<PredictionResult> {
        // Try regular memory first
        if let Some((label, confidence)) = self.memory.retrieve(input) {
            // Get alternatives from continual memory
            let mut alternatives = Vec::new();
            if let Some((alt_label, alt_confidence)) = self.continual_memory.retrieve(input) {
                if alt_label != label {
                    alternatives.push((alt_label, alt_confidence));
                }
            }

            return Ok(PredictionResult {
                predicted_label: label,
                confidence,
                alternatives,
            });
        }

        // Try continual memory
        if let Some((label, confidence)) = self.continual_memory.retrieve(input) {
            return Ok(PredictionResult {
                predicted_label: label,
                confidence,
                alternatives: Vec::new(),
            });
        }

        // No match found
        Ok(PredictionResult {
            predicted_label: "unknown".to_string(),
            confidence: 0.0,
            alternatives: Vec::new(),
        })
    }

    /// Update the system with feedback
    ///
    /// # Arguments
    ///
    /// * `input` - Input hypervector
    /// * `correct_label` - Correct label for the input
    /// * `learning_rate` - Learning rate for this update
    /// * `prediction_error` - Error in the prediction
    ///
    /// # Returns
    ///
    /// Update result information
    pub fn update_with_feedback(
        &mut self,
        input: &Hypervector,
        correct_label: &str,
        learning_rate: f64,
        prediction_error: f64,
    ) -> NdimageResult<UpdateResult> {
        // Record performance
        let accuracy = 1.0 - prediction_error;
        self.performance_tracker
            .record_update(accuracy, learning_rate);

        // Update memory
        self.memory
            .update_pattern(correct_label.to_string(), input.clone(), learning_rate)?;

        // Create experience for continual learning
        let experience = Experience {
            encoding: input.clone(),
            label: correct_label.to_string(),
            timestamp: self.performance_tracker.update_count,
            importance: 1.0 - prediction_error,
        };

        let consolidation = ConsolidationResult {
            interference_prevented: 0,
            effectiveness_score: accuracy,
            replay_cycles_used: 1,
        };

        self.continual_memory
            .add_experience(experience, &consolidation)?;

        // Adapt learning parameters
        self.adaptation_params
            .adjust_based_on_performance(&self.performance_tracker);

        // Update learning state based on performance
        self.update_learning_state();

        Ok(UpdateResult {
            memory_updated: true,
            learning_rate_used: learning_rate,
            performance_change: self.performance_tracker.get_recent_performance_change(),
        })
    }

    /// Perform a complete online learning step
    ///
    /// # Arguments
    ///
    /// * `input` - Input hypervector
    /// * `true_label` - True label (if available for supervised learning)
    ///
    /// # Returns
    ///
    /// Complete learning result
    pub fn online_learning_step(
        &mut self,
        input: &Hypervector,
        true_label: Option<&str>,
    ) -> NdimageResult<OnlineLearningResult> {
        // Make prediction
        let prediction = self.predict(input)?;

        // Calculate error and update if true label is provided
        let learning_update = if let Some(label) = true_label {
            let error = calculate_prediction_error(&prediction, label);
            self.update_with_feedback(input, label, self.adaptation_params.current_rate, error)?
        } else {
            UpdateResult {
                memory_updated: false,
                learning_rate_used: 0.0,
                performance_change: 0.0,
            }
        };

        // Get current performance metrics
        let system_performance = self.get_performancemetrics();

        Ok(OnlineLearningResult {
            prediction,
            learning_update,
            system_performance,
            adaptation_rate: self.adaptation_params.current_rate,
        })
    }

    /// Update learning state based on current performance
    fn update_learning_state(&mut self) {
        let recent_change = self.performance_tracker.get_recent_performance_change();
        let accuracy = self.performance_tracker.get_accuracy();

        self.learning_state = if recent_change < -0.1 && accuracy < 0.7 {
            LearningState::RapidAdaptation
        } else if accuracy > 0.9 && recent_change.abs() < 0.05 {
            LearningState::Conservative
        } else {
            LearningState::Normal
        };
    }

    /// Get current performance metrics
    pub fn get_performancemetrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            accuracy: self.performance_tracker.get_accuracy(),
            learning_speed: self.performance_tracker.get_learning_speed(),
            memory_efficiency: self.performance_tracker.get_memory_efficiency(),
            adaptation_effectiveness: self.adaptation_params.current_rate
                / self.adaptation_params.base_rate,
        }
    }

    /// Perform maintenance cycle (cleanup, consolidation, etc.)
    pub fn perform_maintenance_cycle(&mut self, _config: &HDCConfig) -> NdimageResult<()> {
        // Simplified maintenance - could implement more sophisticated cleanup

        // Reset adaptation if performance is poor for too long
        if self.performance_tracker.get_accuracy() < 0.5
            && self.performance_tracker.update_count > 100
        {
            self.adaptation_params.reset();
        }

        // Could implement more maintenance tasks:
        // - Memory consolidation
        // - Pattern cleanup
        // - Interference detection and resolution

        Ok(())
    }

    /// Get the current learning state
    pub fn get_learning_state(&self) -> &LearningState {
        &self.learning_state
    }

    /// Get memory statistics
    pub fn get_memory_stats(&self) -> (usize, (usize, usize, usize)) {
        (self.memory.size(), self.continual_memory.get_memory_stats())
    }

    /// Compute adaptive learning rate based on prediction error
    pub fn compute_adaptive_learning_rate(&self, prediction_error: f64) -> f64 {
        // Simple adaptive learning rate based on error
        let base_rate = self.adaptation_params.base_rate;
        let error_factor = 1.0 + prediction_error;
        (base_rate * error_factor)
            .min(self.adaptation_params.max_rate)
            .max(self.adaptation_params.min_rate)
    }

    /// Perform unsupervised update
    pub fn unsupervised_update(&mut self, input: &Hypervector) -> NdimageResult<UpdateResult> {
        // Simple unsupervised learning - just store the pattern
        let synthetic_label = format!("unsupervised_{}", self.performance_tracker.update_count);
        self.memory.store(synthetic_label, input.clone());

        Ok(UpdateResult {
            memory_updated: true,
            learning_rate_used: self.adaptation_params.current_rate,
            performance_change: 0.0, // No feedback available
        })
    }

    /// Get performance metrics (renamed method)
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.get_performancemetrics()
    }

    /// Get current adaptation rate
    pub fn get_current_adaptation_rate(&self) -> f64 {
        self.adaptation_params.current_rate
    }
}

/// Calculate prediction error
///
/// # Arguments
///
/// * `prediction` - The prediction result
/// * `true_label` - The correct label
///
/// # Returns
///
/// Error value between 0.0 and 1.0
pub fn calculate_prediction_error(prediction: &PredictionResult, true_label: &str) -> f64 {
    if prediction.predicted_label == true_label {
        0.0 // Perfect prediction
    } else {
        1.0 - prediction.confidence // Error based on confidence in wrong prediction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperdimensional_computing::vector_ops::*;

    #[test]
    fn test_hdc_memory_basic_operations() {
        let config = HDCConfig::default();
        let mut memory = HDCMemory::new(config);

        let hv1 = Hypervector::random(1000, 0.1);
        let hv2 = Hypervector::random(1000, 0.1);

        // Store patterns
        memory.store("pattern1".to_string(), hv1.clone());
        memory.store("pattern2".to_string(), hv2.clone());

        assert_eq!(memory.size(), 2);

        // Test retrieval
        let result = memory.retrieve(&hv1);
        assert!(result.is_some());
        let (label, confidence) = result.unwrap();
        assert_eq!(label, "pattern1");
        assert!(confidence > 0.8);

        // Test removal
        let removed = memory.remove("pattern1");
        assert!(removed.is_some());
        assert_eq!(memory.size(), 1);

        // Test clear
        memory.clear();
        assert_eq!(memory.size(), 0);
    }

    #[test]
    fn test_continual_learning_memory() {
        let config = HDCConfig::default();
        let mut memory = ContinualLearningMemory::new(&config);

        let encoding = Hypervector::random(config.hypervector_dim, config.sparsity);
        let experience = Experience {
            encoding: encoding.clone(),
            label: "test".to_string(),
            timestamp: 0,
            importance: 0.8,
        };

        let consolidation = ConsolidationResult {
            interference_prevented: 1,
            effectiveness_score: 0.9,
            replay_cycles_used: 3,
        };

        assert!(memory.add_experience(experience, &consolidation).is_ok());
        assert_eq!(memory.episodic_buffer.len(), 1);

        let interference = memory.calculate_interference(&encoding);
        assert!(interference >= 0.0);
        assert!(interference <= 1.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new();

        // Record some updates
        tracker.record_update(0.8, 0.1);
        tracker.record_update(0.9, 0.12);
        tracker.record_update(0.85, 0.11);

        assert_eq!(tracker.update_count, 3);

        let accuracy = tracker.get_accuracy();
        assert!(accuracy > 0.8);
        assert!(accuracy < 0.9);

        let learning_speed = tracker.get_learning_speed();
        assert!(learning_speed > 0.1);
        assert!(learning_speed < 0.13);

        let memory_efficiency = tracker.get_memory_efficiency();
        assert!(memory_efficiency > 0.0);
        assert!(memory_efficiency <= 1.0);

        // Test reset
        tracker.reset();
        assert_eq!(tracker.update_count, 0);
        assert!(tracker.accuracyhistory.is_empty());
    }

    #[test]
    fn test_online_learning_system() {
        let config = HDCConfig::default();
        let mut system = OnlineLearningSystem::new(&config);

        let encoding = Hypervector::random(config.hypervector_dim, config.sparsity);

        // Test prediction on empty system
        let prediction = system.predict(&encoding).unwrap();
        assert_eq!(prediction.predicted_label, "unknown");
        assert_eq!(prediction.confidence, 0.0);

        // Test update with feedback
        let learning_rate = 0.1;
        let error = 0.5;
        let update_result = system
            .update_with_feedback(&encoding, "test_label", learning_rate, error)
            .unwrap();
        assert!(update_result.memory_updated);
        assert_eq!(update_result.learning_rate_used, learning_rate);

        // Test online learning step
        let result = system
            .online_learning_step(&encoding, Some("test_label"))
            .unwrap();
        assert!(result.prediction.confidence > 0.0);
        assert!(result.system_performance.accuracy > 0.0);

        // Test maintenance cycle
        assert!(system.perform_maintenance_cycle(&config).is_ok());

        // Test performance metrics
        let metrics = system.get_performancemetrics();
        assert!(metrics.accuracy >= 0.0);
        assert!(metrics.accuracy <= 1.0);
    }

    #[test]
    fn test_calculate_prediction_error() {
        let correct_prediction = PredictionResult {
            predicted_label: "cat".to_string(),
            confidence: 0.9,
            alternatives: Vec::new(),
        };

        let incorrect_prediction = PredictionResult {
            predicted_label: "dog".to_string(),
            confidence: 0.7,
            alternatives: Vec::new(),
        };

        let error1 = calculate_prediction_error(&correct_prediction, "cat");
        assert_eq!(error1, 0.0);

        let error2 = calculate_prediction_error(&incorrect_prediction, "cat");
        assert_eq!(error2, 1.0 - 0.7); // 1.0 - confidence
    }

    #[test]
    fn test_memory_update_pattern() {
        let config = HDCConfig::default();
        let mut memory = HDCMemory::new(config);

        let hv1 = Hypervector::random(1000, 0.1);
        let hv2 = Hypervector::random(1000, 0.1);

        // Store initial pattern
        memory.store("test".to_string(), hv1.clone());

        // Update with new pattern
        assert!(memory.update_pattern("test".to_string(), hv2, 0.5).is_ok());

        // Pattern should be updated
        let stored = memory.patterns.get("test").unwrap();
        assert!(stored.similarity(&hv1) < 1.0); // Should be different from original
    }
}
