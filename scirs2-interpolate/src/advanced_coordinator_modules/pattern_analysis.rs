//! Data pattern analysis system for interpolation optimization
//!
//! This module provides sophisticated pattern recognition and analysis capabilities
//! for automatically identifying optimal interpolation methods based on data characteristics.

use super::types::*;
use crate::error::InterpolateResult;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Data pattern analyzer for intelligent interpolation
#[derive(Debug)]
pub struct DataPatternAnalyzer<F: Float + Debug> {
    /// Pattern database
    pattern_db: HashMap<PatternSignature, PatternData<F>>,
    /// Current analysis state
    analysis_state: AnalysisState<F>,
    /// Pattern recognition model
    recognition_model: PatternRecognitionModel<F>,
}

/// Pattern signature for identification and classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct PatternSignature {
    /// Pattern type
    pattern_type: DataPatternType,
    /// Size characteristics
    size_range: (usize, usize),
    /// Smoothness characteristics
    smoothness_profile: SmoothnessProfile,
}

/// Smoothness profile classification
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum SmoothnessProfile {
    /// Very smooth (C∞)
    VerySmooth,
    /// Smooth (C²)
    Smooth,
    /// Moderately smooth (C¹)
    ModeratelySmooth,
    /// Continuous (C⁰)
    Continuous,
    /// Discontinuous
    Discontinuous,
}

/// Analysis state for pattern recognition
#[derive(Debug)]
pub struct AnalysisState<F: Float> {
    /// Current data being analyzed
    current_data: Option<DataProfile<F>>,
    /// Analysis progress (0-1)
    progress: f64,
    /// Intermediate results
    intermediate_results: HashMap<String, f64>,
    /// Current analysis stage
    current_stage: AnalysisStage,
}

/// Pattern recognition model
#[derive(Debug)]
pub struct PatternRecognitionModel<F: Float> {
    /// Feature extractors
    feature_extractors: Vec<FeatureExtractor<F>>,
    /// Classification weights
    classification_weights: HashMap<String, f64>,
    /// Model accuracy
    model_accuracy: f64,
}

/// Feature extractor for pattern recognition
#[derive(Debug)]
pub struct FeatureExtractor<F: Float> {
    /// Feature name
    pub name: String,
    /// Feature extraction function
    pub extractor: fn(&[F]) -> f64,
    /// Feature importance weight
    pub importance: f64,
}

/// Pattern analysis result
#[derive(Debug, Clone)]
pub struct PatternAnalysisResult<F: Float> {
    /// Identified pattern signature
    pub pattern_signature: PatternSignature,
    /// Confidence in pattern identification (0-1)
    pub confidence: f64,
    /// Extracted features
    pub features: HashMap<String, f64>,
    /// Recommended interpolation methods
    pub recommended_methods: Vec<InterpolationMethodType>,
    /// Performance predictions
    pub performance_predictions: HashMap<InterpolationMethodType, PerformanceCharacteristics>,
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata<F>,
}

/// Metadata about the analysis process
#[derive(Debug, Clone)]
pub struct AnalysisMetadata<F: Float> {
    /// Time taken for analysis
    pub analysis_time_ms: f64,
    /// Data quality score
    pub data_quality_score: F,
    /// Number of features extracted
    pub feature_count: usize,
    /// Model version used
    pub model_version: String,
}

impl<F: Float + Debug> DataPatternAnalyzer<F> {
    /// Create a new data pattern analyzer
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            pattern_db: HashMap::new(),
            analysis_state: AnalysisState::new()?,
            recognition_model: PatternRecognitionModel::new()?,
        })
    }

    /// Analyze data patterns and recommend optimal interpolation methods
    pub fn analyze_patterns(
        &mut self,
        data_profile: &DataProfile<F>,
    ) -> InterpolateResult<PatternAnalysisResult<F>> {
        let start_time = std::time::Instant::now();

        // Update analysis state
        self.analysis_state.current_data = Some(data_profile.clone());
        self.analysis_state.progress = 0.0;
        self.analysis_state.current_stage = AnalysisStage::DataLoading;

        // Extract features from data profile
        self.analysis_state.current_stage = AnalysisStage::FeatureExtraction;
        let features = self.extract_features(data_profile)?;
        self.analysis_state.progress = 0.3;

        // Identify pattern signature
        self.analysis_state.current_stage = AnalysisStage::PatternRecognition;
        let pattern_signature = self.identify_pattern_signature(data_profile, &features)?;
        self.analysis_state.progress = 0.6;

        // Generate method recommendations
        self.analysis_state.current_stage = AnalysisStage::MethodRecommendation;
        let recommended_methods = self.recommend_methods(&pattern_signature, &features)?;
        self.analysis_state.progress = 0.8;

        // Predict performance for each method
        self.analysis_state.current_stage = AnalysisStage::PerformancePrediction;
        let performance_predictions =
            self.predict_performance(&recommended_methods, data_profile, &features)?;
        self.analysis_state.progress = 0.9;

        // Calculate confidence
        let confidence = self.calculate_pattern_confidence(&pattern_signature, &features);

        // Create analysis metadata
        let analysis_time = start_time.elapsed().as_millis() as f64;
        let metadata = AnalysisMetadata {
            analysis_time_ms: analysis_time,
            data_quality_score: self.assess_data_quality(data_profile),
            feature_count: features.len(),
            model_version: "1.0.0".to_string(),
        };

        self.analysis_state.current_stage = AnalysisStage::Completed;
        self.analysis_state.progress = 1.0;

        Ok(PatternAnalysisResult {
            pattern_signature,
            confidence,
            features,
            recommended_methods,
            performance_predictions,
            analysis_metadata: metadata,
        })
    }

    /// Extract features from data profile
    fn extract_features(
        &self,
        data_profile: &DataProfile<F>,
    ) -> InterpolateResult<HashMap<String, f64>> {
        let mut features = HashMap::new();

        // Basic statistical features
        features.insert("data_size".to_string(), data_profile.size as f64);
        features.insert(
            "dimensionality".to_string(),
            data_profile.dimensionality as f64,
        );
        features.insert(
            "smoothness".to_string(),
            data_profile.smoothness.to_f64().unwrap_or(0.5),
        );
        features.insert(
            "noise_level".to_string(),
            data_profile.noise_level.to_f64().unwrap_or(0.1),
        );
        features.insert(
            "sparsity".to_string(),
            data_profile.sparsity.to_f64().unwrap_or(0.0),
        );

        // Value range features
        let value_range = data_profile.value_range.1 - data_profile.value_range.0;
        features.insert(
            "value_range".to_string(),
            value_range.to_f64().unwrap_or(1.0),
        );

        // Gradient features
        features.insert(
            "mean_gradient".to_string(),
            data_profile
                .gradient_stats
                .mean_magnitude
                .to_f64()
                .unwrap_or(0.0),
        );
        features.insert(
            "gradient_variance".to_string(),
            data_profile.gradient_stats.variance.to_f64().unwrap_or(0.0),
        );
        features.insert(
            "max_gradient".to_string(),
            data_profile
                .gradient_stats
                .max_gradient
                .to_f64()
                .unwrap_or(0.0),
        );

        // Frequency features
        features.insert(
            "dominant_frequency".to_string(),
            data_profile
                .frequency_content
                .dominant_frequency
                .to_f64()
                .unwrap_or(0.0),
        );
        features.insert(
            "high_freq_ratio".to_string(),
            data_profile
                .frequency_content
                .high_freq_ratio
                .to_f64()
                .unwrap_or(0.0),
        );
        features.insert(
            "low_freq_ratio".to_string(),
            data_profile
                .frequency_content
                .low_freq_ratio
                .to_f64()
                .unwrap_or(0.0),
        );

        // Derived features
        let complexity_score = self.calculate_complexity_score(data_profile);
        features.insert("complexity_score".to_string(), complexity_score);

        let regularity_score = self.calculate_regularity_score(data_profile);
        features.insert("regularity_score".to_string(), regularity_score);

        let interpolation_difficulty = self.calculate_interpolation_difficulty(data_profile);
        features.insert(
            "interpolation_difficulty".to_string(),
            interpolation_difficulty,
        );

        Ok(features)
    }

    /// Identify pattern signature based on data characteristics
    fn identify_pattern_signature(
        &self,
        data_profile: &DataProfile<F>,
        features: &HashMap<String, f64>,
    ) -> InterpolateResult<PatternSignature> {
        // Classify pattern type
        let pattern_type = self.classify_pattern_type(features);

        // Determine size range
        let size_range = match data_profile.size {
            0..=1000 => (0, 1000),
            1001..=10000 => (1001, 10000),
            10001..=100000 => (10001, 100000),
            _ => (100001, usize::MAX),
        };

        // Classify smoothness profile
        let smoothness_profile = self.classify_smoothness_profile(features);

        Ok(PatternSignature {
            pattern_type,
            size_range,
            smoothness_profile,
        })
    }

    /// Classify data pattern type based on features
    fn classify_pattern_type(&self, features: &HashMap<String, f64>) -> DataPatternType {
        let smoothness = features.get("smoothness").copied().unwrap_or(0.5);
        let noise_level = features.get("noise_level").copied().unwrap_or(0.1);
        let sparsity = features.get("sparsity").copied().unwrap_or(0.0);
        let high_freq_ratio = features.get("high_freq_ratio").copied().unwrap_or(0.0);
        let regularity = features.get("regularity_score").copied().unwrap_or(0.5);

        if noise_level > 0.2 {
            DataPatternType::Noisy
        } else if sparsity > 0.3 {
            DataPatternType::Sparse
        } else if high_freq_ratio > 0.6 {
            DataPatternType::Oscillatory
        } else if smoothness > 0.8 && regularity > 0.7 {
            DataPatternType::Smooth
        } else if regularity > 0.8 {
            DataPatternType::Structured
        } else if smoothness < 0.3 {
            DataPatternType::Irregular
        } else {
            // Check for monotonicity
            let gradient_variance = features.get("gradient_variance").copied().unwrap_or(1.0);
            if gradient_variance < 0.1 {
                DataPatternType::Monotonic
            } else {
                DataPatternType::PiecewiseContinuous
            }
        }
    }

    /// Classify smoothness profile
    fn classify_smoothness_profile(&self, features: &HashMap<String, f64>) -> SmoothnessProfile {
        let smoothness = features.get("smoothness").copied().unwrap_or(0.5);
        let gradient_variance = features.get("gradient_variance").copied().unwrap_or(1.0);
        let max_gradient = features.get("max_gradient").copied().unwrap_or(1.0);

        if smoothness >= 0.95 && gradient_variance < 0.01 {
            SmoothnessProfile::VerySmooth
        } else if smoothness >= 0.8 && gradient_variance < 0.1 {
            SmoothnessProfile::Smooth
        } else if smoothness >= 0.6 && max_gradient < 10.0 {
            SmoothnessProfile::ModeratelySmooth
        } else if smoothness >= 0.3 {
            SmoothnessProfile::Continuous
        } else {
            SmoothnessProfile::Discontinuous
        }
    }

    /// Recommend interpolation methods based on pattern
    fn recommend_methods(
        &self,
        pattern_signature: &PatternSignature,
        features: &HashMap<String, f64>,
    ) -> InterpolateResult<Vec<InterpolationMethodType>> {
        let mut methods = Vec::new();
        let data_size = features.get("data_size").copied().unwrap_or(1000.0) as usize;
        let dimensionality = features.get("dimensionality").copied().unwrap_or(1.0) as usize;

        match &pattern_signature.pattern_type {
            DataPatternType::Smooth => {
                methods.push(InterpolationMethodType::CubicSpline);
                methods.push(InterpolationMethodType::BSpline);
                if data_size < 5000 {
                    methods.push(InterpolationMethodType::Polynomial);
                }
            }
            DataPatternType::Noisy => {
                methods.push(InterpolationMethodType::BSpline);
                methods.push(InterpolationMethodType::Linear);
                if dimensionality <= 3 {
                    methods.push(InterpolationMethodType::Kriging);
                }
            }
            DataPatternType::Oscillatory => {
                methods.push(InterpolationMethodType::AkimaSpline);
                methods.push(InterpolationMethodType::PchipInterpolation);
                methods.push(InterpolationMethodType::BSpline);
            }
            DataPatternType::Sparse => {
                methods.push(InterpolationMethodType::RadialBasisFunction);
                methods.push(InterpolationMethodType::NaturalNeighbor);
                if dimensionality <= 3 {
                    methods.push(InterpolationMethodType::ThinPlateSpline);
                }
            }
            DataPatternType::Irregular => {
                methods.push(InterpolationMethodType::RadialBasisFunction);
                methods.push(InterpolationMethodType::ShepardsMethod);
                methods.push(InterpolationMethodType::NaturalNeighbor);
            }
            DataPatternType::Monotonic => {
                methods.push(InterpolationMethodType::PchipInterpolation);
                methods.push(InterpolationMethodType::AkimaSpline);
                methods.push(InterpolationMethodType::CubicSpline);
            }
            DataPatternType::Structured => {
                methods.push(InterpolationMethodType::BSpline);
                methods.push(InterpolationMethodType::CubicSpline);
                if data_size > 1000 {
                    methods.push(InterpolationMethodType::Linear);
                }
            }
            DataPatternType::PiecewiseContinuous => {
                methods.push(InterpolationMethodType::Linear);
                methods.push(InterpolationMethodType::PchipInterpolation);
                methods.push(InterpolationMethodType::BSpline);
            }
        }

        // Add quantum-inspired method for complex patterns
        let complexity = features.get("complexity_score").copied().unwrap_or(0.5);
        if complexity > 0.8 && data_size < 10000 {
            methods.push(InterpolationMethodType::QuantumInspired);
        }

        // Ensure we have at least some methods
        if methods.is_empty() {
            methods.push(InterpolationMethodType::Linear);
            methods.push(InterpolationMethodType::CubicSpline);
        }

        // Limit to top 5 methods
        methods.truncate(5);

        Ok(methods)
    }

    /// Predict performance for recommended methods
    fn predict_performance(
        &self,
        methods: &[InterpolationMethodType],
        data_profile: &DataProfile<F>,
        features: &HashMap<String, f64>,
    ) -> InterpolateResult<HashMap<InterpolationMethodType, PerformanceCharacteristics>> {
        let mut predictions = HashMap::new();

        for &method in methods {
            let characteristics =
                self.predict_method_performance(method, data_profile, features)?;
            predictions.insert(method, characteristics);
        }

        Ok(predictions)
    }

    /// Predict performance characteristics for a specific method
    fn predict_method_performance(
        &self,
        method: InterpolationMethodType,
        data_profile: &DataProfile<F>,
        features: &HashMap<String, f64>,
    ) -> InterpolateResult<PerformanceCharacteristics> {
        let data_size = data_profile.size;
        let complexity = features.get("complexity_score").copied().unwrap_or(0.5);
        let noise_level = features.get("noise_level").copied().unwrap_or(0.1);

        let (base_time, base_memory, base_accuracy, base_robustness) = match method {
            InterpolationMethodType::Linear => (1.0, 1.0, 0.7, 0.9),
            InterpolationMethodType::CubicSpline => (2.0, 1.5, 0.9, 0.7),
            InterpolationMethodType::BSpline => (3.0, 2.0, 0.92, 0.8),
            InterpolationMethodType::RadialBasisFunction => (10.0, 5.0, 0.95, 0.6),
            InterpolationMethodType::Kriging => (15.0, 8.0, 0.98, 0.9),
            InterpolationMethodType::Polynomial => (5.0, 3.0, 0.85, 0.5),
            InterpolationMethodType::PchipInterpolation => (2.5, 1.8, 0.88, 0.85),
            InterpolationMethodType::AkimaSpline => (2.8, 2.2, 0.87, 0.82),
            InterpolationMethodType::ThinPlateSpline => (12.0, 6.0, 0.93, 0.75),
            InterpolationMethodType::NaturalNeighbor => (8.0, 4.0, 0.86, 0.8),
            InterpolationMethodType::ShepardsMethod => (6.0, 3.0, 0.75, 0.7),
            InterpolationMethodType::QuantumInspired => (20.0, 10.0, 0.99, 0.95),
        };

        // Adjust for data size
        let size_factor = (data_size as f64).log10() / 3.0; // Normalize to log10(1000)
        let time_multiplier = base_time * (1.0 + size_factor * complexity);
        let memory_multiplier = base_memory * (1.0 + size_factor * 0.5);

        // Adjust accuracy for noise
        let accuracy_penalty = noise_level * 0.3;
        let expected_accuracy = (base_accuracy - accuracy_penalty).max(0.1);

        // Adjust robustness
        let robustness_bonus = if noise_level > 0.1 { 0.1 } else { 0.0 };
        let robustness_score = (base_robustness + robustness_bonus).min(1.0);

        Ok(PerformanceCharacteristics {
            throughput: 1.0 / time_multiplier.max(0.1), // Convert time multiplier to throughput
            memory_efficiency: 1.0 / memory_multiplier.max(0.1), // Convert memory multiplier to efficiency
            accuracy_score: expected_accuracy,
            stability: robustness_score,
            scalability: 1.0, // Default scalability value
        })
    }

    /// Calculate pattern identification confidence
    fn calculate_pattern_confidence(
        &self,
        _pattern_signature: &PatternSignature,
        features: &HashMap<String, f64>,
    ) -> f64 {
        // Simple confidence calculation based on feature clarity
        let smoothness = features.get("smoothness").copied().unwrap_or(0.5);
        let noise_level = features.get("noise_level").copied().unwrap_or(0.1);
        let regularity = features.get("regularity_score").copied().unwrap_or(0.5);

        // Higher confidence for clear patterns
        let clarity_score = (smoothness + regularity) / 2.0;
        let noise_penalty = noise_level * 0.5;

        ((clarity_score - noise_penalty) * self.recognition_model.model_accuracy)
            .max(0.1)
            .min(1.0)
    }

    /// Assess data quality
    fn assess_data_quality(&self, data_profile: &DataProfile<F>) -> F {
        let noise_factor = F::one() - data_profile.noise_level.min(F::one());
        let sparsity_factor = F::one() - data_profile.sparsity.min(F::one());
        let size_factor = if data_profile.size > 100 {
            F::one()
        } else {
            F::from(data_profile.size as f64 / 100.0).unwrap()
        };

        (noise_factor + sparsity_factor + size_factor) / F::from(3.0).unwrap()
    }

    /// Calculate complexity score for data
    fn calculate_complexity_score(&self, data_profile: &DataProfile<F>) -> f64 {
        let gradient_complexity = data_profile.gradient_stats.variance.to_f64().unwrap_or(0.0);
        let frequency_complexity = data_profile
            .frequency_content
            .frequency_spread
            .to_f64()
            .unwrap_or(0.0);
        let dimensional_complexity = (data_profile.dimensionality as f64).log10();

        (gradient_complexity + frequency_complexity + dimensional_complexity) / 3.0
    }

    /// Calculate regularity score
    fn calculate_regularity_score(&self, data_profile: &DataProfile<F>) -> f64 {
        let smoothness = data_profile.smoothness.to_f64().unwrap_or(0.5);
        let low_freq_dominance = data_profile
            .frequency_content
            .low_freq_ratio
            .to_f64()
            .unwrap_or(0.5);
        let gradient_consistency =
            1.0 / (1.0 + data_profile.gradient_stats.variance.to_f64().unwrap_or(1.0));

        (smoothness + low_freq_dominance + gradient_consistency) / 3.0
    }

    /// Calculate interpolation difficulty
    fn calculate_interpolation_difficulty(&self, data_profile: &DataProfile<F>) -> f64 {
        let noise_difficulty = data_profile.noise_level.to_f64().unwrap_or(0.1);
        let sparsity_difficulty = data_profile.sparsity.to_f64().unwrap_or(0.0);
        let irregularity_difficulty = 1.0 - data_profile.smoothness.to_f64().unwrap_or(0.5);
        let size_difficulty = if data_profile.size > 100000 { 0.3 } else { 0.0 };

        (noise_difficulty + sparsity_difficulty + irregularity_difficulty + size_difficulty) / 4.0
    }

    /// Get current analysis state
    pub fn get_analysis_state(&self) -> &AnalysisState<F> {
        &self.analysis_state
    }

    /// Get pattern database statistics
    pub fn get_pattern_statistics(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        stats.insert("total_patterns".to_string(), self.pattern_db.len());

        // Count patterns by type
        let mut type_counts = HashMap::new();
        for signature in self.pattern_db.keys() {
            let pattern_name = format!("{:?}", signature.pattern_type);
            *type_counts.entry(pattern_name).or_insert(0) += 1;
        }

        for (pattern_type, count) in type_counts {
            stats.insert(pattern_type, count);
        }

        stats
    }

    /// Add pattern to database
    pub fn add_pattern(&mut self, signature: PatternSignature, data: PatternData<F>) {
        self.pattern_db.insert(signature, data);
    }

    /// Update recognition model
    pub fn update_recognition_model(
        &mut self,
        feedback: &[(PatternSignature, bool)],
    ) -> InterpolateResult<()> {
        if feedback.len() >= 10 {
            let success_rate = feedback.iter().filter(|(_, success)| *success).count() as f64
                / feedback.len() as f64;
            self.recognition_model.model_accuracy = success_rate * 0.9 + 0.1; // Add baseline confidence
        }
        Ok(())
    }
}

impl<F: Float> AnalysisState<F> {
    /// Create a new analysis state
    pub fn new() -> InterpolateResult<Self> {
        Ok(Self {
            current_data: None,
            progress: 0.0,
            intermediate_results: HashMap::new(),
            current_stage: AnalysisStage::DataLoading,
        })
    }

    /// Get current analysis progress (0-1)
    pub fn get_progress(&self) -> f64 {
        self.progress
    }

    /// Get current analysis stage
    pub fn get_current_stage(&self) -> &AnalysisStage {
        &self.current_stage
    }

    /// Get intermediate results
    pub fn get_intermediate_results(&self) -> &HashMap<String, f64> {
        &self.intermediate_results
    }
}

impl<F: Float> PatternRecognitionModel<F> {
    /// Create a new pattern recognition model
    pub fn new() -> InterpolateResult<Self> {
        let mut model = Self {
            feature_extractors: Vec::new(),
            classification_weights: HashMap::new(),
            model_accuracy: 0.8, // Initial confidence
        };

        // Initialize with default feature extractors
        model.initialize_default_extractors();

        Ok(model)
    }

    /// Initialize with default feature extractors
    fn initialize_default_extractors(&mut self) {
        // Statistical features
        self.feature_extractors.push(FeatureExtractor {
            name: "mean".to_string(),
            extractor: |data| {
                data.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64
            },
            importance: 0.6,
        });

        self.feature_extractors.push(FeatureExtractor {
            name: "variance".to_string(),
            extractor: |data| {
                let mean =
                    data.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum::<f64>() / data.len() as f64;
                data.iter()
                    .map(|x| {
                        let diff = x.to_f64().unwrap_or(0.0) - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / data.len() as f64
            },
            importance: 0.8,
        });

        // Add classification weights
        self.classification_weights
            .insert("smoothness_weight".to_string(), 0.4);
        self.classification_weights
            .insert("noise_weight".to_string(), 0.3);
        self.classification_weights
            .insert("complexity_weight".to_string(), 0.3);
    }

    /// Get model accuracy
    pub fn get_model_accuracy(&self) -> f64 {
        self.model_accuracy
    }

    /// Update model accuracy
    pub fn update_accuracy(&mut self, accuracy: f64) {
        self.model_accuracy = accuracy.max(0.0).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_analyzer_creation() {
        let analyzer: DataPatternAnalyzer<f64> = DataPatternAnalyzer::new().unwrap();
        assert_eq!(analyzer.pattern_db.len(), 0);
        assert_eq!(analyzer.analysis_state.progress, 0.0);
    }

    #[test]
    fn test_smoothness_profile_classification() {
        let analyzer: DataPatternAnalyzer<f64> = DataPatternAnalyzer::new().unwrap();
        let mut features = HashMap::new();

        features.insert("smoothness".to_string(), 0.95);
        features.insert("gradient_variance".to_string(), 0.005);
        let profile = analyzer.classify_smoothness_profile(&features);
        assert!(matches!(profile, SmoothnessProfile::VerySmooth));

        features.insert("smoothness".to_string(), 0.2);
        let profile = analyzer.classify_smoothness_profile(&features);
        assert!(matches!(profile, SmoothnessProfile::Discontinuous));
    }

    #[test]
    fn test_pattern_type_classification() {
        let analyzer: DataPatternAnalyzer<f64> = DataPatternAnalyzer::new().unwrap();
        let mut features = HashMap::new();

        // Test noisy pattern
        features.insert("noise_level".to_string(), 0.3);
        features.insert("smoothness".to_string(), 0.5);
        features.insert("sparsity".to_string(), 0.1);
        let pattern = analyzer.classify_pattern_type(&features);
        assert!(matches!(pattern, DataPatternType::Noisy));

        // Test smooth pattern
        features.insert("noise_level".to_string(), 0.05);
        features.insert("smoothness".to_string(), 0.9);
        features.insert("regularity_score".to_string(), 0.8);
        let pattern = analyzer.classify_pattern_type(&features);
        assert!(matches!(pattern, DataPatternType::Smooth));
    }
}
