//! Spike pattern recognition for neuromorphic systems
//!
//! This module provides pattern recognition capabilities for detecting
//! and analyzing spike patterns in neuromorphic networks.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use num_traits::Float;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Spike pattern recognizer
#[derive(Debug)]
pub struct SpikePatternRecognizer<F: Float> {
    /// Pattern templates
    pub pattern_templates: Vec<SpikePattern<F>>,
    /// Recognition thresholds
    pub thresholds: HashMap<String, F>,
    /// Pattern matching algorithms
    pub matching_algorithms: Vec<PatternMatchingAlgorithm>,
    /// Recognition history
    pub recognition_history: VecDeque<PatternRecognition<F>>,
}

/// Spike pattern template
#[derive(Debug, Clone)]
pub struct SpikePattern<F: Float> {
    /// Pattern name
    pub name: String,
    /// Spatial pattern (which neurons)
    pub spatial_pattern: Vec<usize>,
    /// Temporal pattern (spike timings)
    pub temporal_pattern: Vec<Duration>,
    /// Pattern strength
    pub strength: F,
    /// Variability tolerance
    pub tolerance: F,
}

/// Pattern matching algorithms
#[derive(Debug, Clone)]
pub enum PatternMatchingAlgorithm {
    /// Cross-correlation based
    CrossCorrelation,
    /// Dynamic time warping
    DynamicTimeWarping,
    /// Hidden Markov models
    HiddenMarkov,
    /// Neural network classifier
    NeuralClassifier,
    /// Template matching
    TemplateMatching,
}

/// Pattern recognition result
#[derive(Debug, Clone)]
pub struct PatternRecognition<F: Float> {
    pub timestamp: Instant,
    pub pattern_name: String,
    pub confidence: F,
    pub matching_neurons: Vec<usize>,
    pub temporal_offset: Duration,
}

impl<F: Float + std::iter::Sum> SpikePatternRecognizer<F> {
    /// Create new pattern recognizer
    pub fn new() -> Self {
        Self {
            pattern_templates: Vec::new(),
            thresholds: HashMap::new(),
            matching_algorithms: vec![PatternMatchingAlgorithm::CrossCorrelation],
            recognition_history: VecDeque::new(),
        }
    }

    /// Add pattern template
    pub fn add_pattern(&mut self, pattern: SpikePattern<F>) {
        self.thresholds
            .insert(pattern.name.clone(), F::from(0.8).unwrap());
        self.pattern_templates.push(pattern);
    }

    /// Recognize patterns in spike data
    pub fn recognize_patterns(
        &mut self,
        spike_data: &HashMap<usize, Vec<Instant>>,
    ) -> crate::error::Result<Vec<PatternRecognition<F>>> {
        let mut recognitions = Vec::new();

        for pattern in &self.pattern_templates {
            if let Some(recognition) = self.match_pattern(pattern, spike_data)? {
                recognitions.push(recognition.clone());
                self.recognition_history.push_back(recognition);
            }
        }

        // Keep bounded
        if self.recognition_history.len() > 1000 {
            self.recognition_history.pop_front();
        }

        Ok(recognitions)
    }

    /// Match a specific pattern
    fn match_pattern(
        &self,
        pattern: &SpikePattern<F>,
        spike_data: &HashMap<usize, Vec<Instant>>,
    ) -> crate::error::Result<Option<PatternRecognition<F>>> {
        let threshold = self
            .thresholds
            .get(&pattern.name)
            .copied()
            .unwrap_or(F::from(0.8).unwrap());

        for algorithm in &self.matching_algorithms {
            let confidence = match algorithm {
                PatternMatchingAlgorithm::CrossCorrelation => {
                    self.cross_correlation_match(pattern, spike_data)?
                }
                PatternMatchingAlgorithm::TemplateMatching => {
                    self.template_match(pattern, spike_data)?
                }
                _ => F::from(0.5).unwrap(), // Default confidence
            };

            if confidence >= threshold {
                return Ok(Some(PatternRecognition {
                    timestamp: Instant::now(),
                    pattern_name: pattern.name.clone(),
                    confidence,
                    matching_neurons: pattern.spatial_pattern.clone(),
                    temporal_offset: Duration::from_millis(0),
                }));
            }
        }

        Ok(None)
    }

    /// Cross-correlation based pattern matching
    fn cross_correlation_match(
        &self,
        pattern: &SpikePattern<F>,
        spike_data: &HashMap<usize, Vec<Instant>>,
    ) -> crate::error::Result<F> {
        let mut correlation_sum = F::zero();
        let mut count = 0;

        for &neuron_id in &pattern.spatial_pattern {
            if let Some(spikes) = spike_data.get(&neuron_id) {
                if !spikes.is_empty() {
                    correlation_sum = correlation_sum + F::one();
                }
                count += 1;
            }
        }

        if count > 0 {
            Ok(correlation_sum / F::from(count).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    /// Template-based pattern matching
    fn template_match(
        &self,
        pattern: &SpikePattern<F>,
        spike_data: &HashMap<usize, Vec<Instant>>,
    ) -> crate::error::Result<F> {
        let mut matched_neurons = 0;
        let total_neurons = pattern.spatial_pattern.len();

        for &neuron_id in &pattern.spatial_pattern {
            if let Some(spikes) = spike_data.get(&neuron_id) {
                if !spikes.is_empty() {
                    matched_neurons += 1;
                }
            }
        }

        if total_neurons > 0 {
            Ok(F::from(matched_neurons).unwrap() / F::from(total_neurons).unwrap())
        } else {
            Ok(F::zero())
        }
    }

    /// Get recognition statistics
    pub fn get_recognition_stats(&self) -> RecognitionStats<F> {
        let total_recognitions = self.recognition_history.len();
        let recent_recognitions = self.recognition_history.iter().rev().take(10).count();

        let avg_confidence = if !self.recognition_history.is_empty() {
            self.recognition_history
                .iter()
                .map(|r| r.confidence)
                .sum::<F>()
                / F::from(self.recognition_history.len()).unwrap()
        } else {
            F::zero()
        };

        RecognitionStats {
            total_recognitions,
            recent_recognitions,
            average_confidence: avg_confidence,
            pattern_counts: self.get_pattern_counts(),
        }
    }

    /// Get pattern recognition counts
    fn get_pattern_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for recognition in &self.recognition_history {
            *counts.entry(recognition.pattern_name.clone()).or_insert(0) += 1;
        }
        counts
    }
}

impl<F: Float> SpikePattern<F> {
    /// Create new spike pattern
    pub fn new(
        name: String,
        spatial_pattern: Vec<usize>,
        temporal_pattern: Vec<Duration>,
        strength: F,
        tolerance: F,
    ) -> Self {
        Self {
            name,
            spatial_pattern,
            temporal_pattern,
            strength,
            tolerance,
        }
    }

    /// Check if pattern matches given spike data
    pub fn matches(&self, spike_data: &HashMap<usize, Vec<Instant>>) -> bool {
        let mut matched_count = 0;

        for &neuron_id in &self.spatial_pattern {
            if let Some(spikes) = spike_data.get(&neuron_id) {
                if !spikes.is_empty() {
                    matched_count += 1;
                }
            }
        }

        let match_ratio = matched_count as f64 / self.spatial_pattern.len() as f64;
        match_ratio >= self.tolerance.to_f64().unwrap_or(0.8)
    }
}

/// Recognition statistics
#[derive(Debug)]
pub struct RecognitionStats<F: Float> {
    pub total_recognitions: usize,
    pub recent_recognitions: usize,
    pub average_confidence: F,
    pub pattern_counts: HashMap<String, usize>,
}
