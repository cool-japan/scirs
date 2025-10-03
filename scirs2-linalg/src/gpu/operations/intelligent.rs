//! Advanced intelligent GPU dispatching with machine learning-based predictions

use super::super::GpuDeviceInfo;
use super::metrics::{MultiDimensionalMetrics, RunningStats};
use crate::error::{LinalgError, LinalgResult};
use scirs2_core::numeric::{Float, NumAssign, Zero};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, Mutex, RwLock};

/// Advanced MODE: Advanced-Intelligent GPU Dispatch System
///
/// This advanced dispatch system uses machine learning-based performance prediction,
/// workload analysis, and adaptive optimization to make optimal CPU/GPU decisions.
pub struct AdvancedIntelligentGpuDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Performance prediction model based on historical data
    performance_predictor: Arc<Mutex<GpuPerformancePredictor>>,
    /// Workload analyzer for operation characteristics
    workload_analyzer: Arc<Mutex<WorkloadAnalyzer>>,
    /// Adaptive threshold optimizer
    threshold_optimizer: Arc<Mutex<AdaptiveThresholdOptimizer>>,
    /// Multi-dimensional performance metrics
    performance_metrics: Arc<RwLock<MultiDimensionalMetrics>>,
    /// Hardware capability profiler
    hardware_profiler: Arc<Mutex<super::hardware::HardwareCapabilityProfiler>>,
    _phantom: std::marker::PhantomData<T>,
}

/// Advanced performance prediction using historical data and workload characteristics
#[derive(Debug)]
pub struct GpuPerformancePredictor {
    /// Historical performance data (operation_signature -> (cpu_time, gpu_time))
    historical_data: HashMap<String, Vec<(f64, f64)>>,
    /// Performance model coefficients
    model_coefficients: HashMap<String, ModelCoefficients>,
    /// Confidence scores for predictions
    confidence_scores: HashMap<String, f64>,
}

/// Model coefficients for performance prediction
#[derive(Debug, Clone)]
pub struct ModelCoefficients {
    /// Matrix size coefficient
    pub size_coeff: f64,
    /// Data type coefficient
    pub dtype_coeff: f64,
    /// Memory bandwidth coefficient
    pub bandwidth_coeff: f64,
    /// Compute intensity coefficient
    pub compute_coeff: f64,
    /// Intercept term
    pub intercept: f64,
}

/// Workload analyzer for understanding operation characteristics
#[derive(Debug)]
pub struct WorkloadAnalyzer {
    /// Matrix sparsity patterns
    sparsity_cache: HashMap<String, f64>,
    /// Memory access patterns
    access_patterns: HashMap<String, MemoryAccessPattern>,
    /// Compute intensity measurements
    compute_intensity: HashMap<String, f64>,
}

/// Memory access pattern characteristics
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided(usize),
    Blocked(usize, usize),
    Hierarchical,
}

/// Adaptive threshold optimizer that learns optimal thresholds
#[derive(Debug)]
pub struct AdaptiveThresholdOptimizer {
    /// Current thresholds for different operations
    current_thresholds: HashMap<String, usize>,
    /// Learning rate for threshold adaptation
    learning_rate: f64,
    /// Performance history for threshold evaluation
    threshold_performance: HashMap<String, VecDeque<(usize, f64, bool)>>, // (threshold, performance, used_gpu)
}

// Supporting data structures
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub sparsity_ratio: f64,
    pub condition_number: Option<f64>,
    pub distribution_type: String,
    pub symmetry: bool,
}

#[derive(Debug)]
pub struct WorkloadAnalysis {
    pub operation: String,
    pub matrixshape: (usize, usize),
    pub compute_intensity: f64,
    pub memory_requirements: usize,
    pub sparsity: f64,
    pub access_pattern: MemoryAccessPattern,
    pub parallelization_potential: f64,
    pub cache_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub device_type: String,
    pub estimated_time: f64,
    pub estimated_energy: f64,
    pub estimated_memory: usize,
    pub confidence_score: f64,
}

#[derive(Debug)]
pub struct OptimalChoice {
    pub selected_device: String,
    pub expected_performance: PerformancePrediction,
    pub optimization_score: f64,
    pub reasoning: String,
}

#[derive(Debug)]
pub struct DispatchDecision {
    pub use_gpu: bool,
    pub selected_device: String,
    pub reasoning: String,
    pub confidence: f64,
    pub estimated_performance: PerformancePrediction,
}

impl<T> AdvancedIntelligentGpuDispatcher<T>
where
    T: Float + NumAssign + Zero + Send + Sync + Debug + 'static,
{
    /// Create a new advanced-intelligent GPU dispatcher
    pub fn new() -> Self {
        Self {
            performance_predictor: Arc::new(Mutex::new(GpuPerformancePredictor::new())),
            workload_analyzer: Arc::new(Mutex::new(WorkloadAnalyzer::new())),
            threshold_optimizer: Arc::new(Mutex::new(AdaptiveThresholdOptimizer::new())),
            performance_metrics: Arc::new(RwLock::new(MultiDimensionalMetrics::new())),
            hardware_profiler: Arc::new(Mutex::new(
                super::hardware::HardwareCapabilityProfiler::new(),
            )),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Make an intelligent dispatch decision using all available information
    pub fn intelligent_dispatch_decision(
        &self,
        operation: &str,
        matrixshape: (usize, usize),
        data_characteristics: &DataCharacteristics,
        available_devices: &[GpuDeviceInfo],
    ) -> LinalgResult<DispatchDecision> {
        // 1. Analyze workload characteristics
        let workload_analysis =
            self.analyze_workload(operation, matrixshape, data_characteristics)?;

        // 2. Predict performance for each option
        let performance_predictions =
            self.predict_performance(operation, &workload_analysis, available_devices)?;

        // 3. Consider multi-dimensional objectives (time, energy, memory)
        let optimal_choice = self.optimize_multi_objective(&performance_predictions)?;

        // 4. Apply adaptive threshold learning
        let final_decision = self.apply_adaptive_thresholds(operation, &optimal_choice)?;

        Ok(final_decision)
    }

    /// Analyze workload characteristics for optimal dispatch
    fn analyze_workload(
        &self,
        operation: &str,
        matrixshape: (usize, usize),
        data_characteristics: &DataCharacteristics,
    ) -> LinalgResult<WorkloadAnalysis> {
        let _analyzer = self.workload_analyzer.lock().map_err(|_| {
            LinalgError::ComputationError("Failed to lock workload analyzer".to_string())
        })?;

        let compute_intensity = self.estimate_compute_intensity(operation, matrixshape);
        let memory_requirements = self.estimate_memory_requirements(matrixshape);
        let sparsity = data_characteristics.sparsity_ratio;
        let access_pattern = self.detect_access_pattern(operation, matrixshape);

        Ok(WorkloadAnalysis {
            operation: operation.to_string(),
            matrixshape,
            compute_intensity,
            memory_requirements,
            sparsity,
            access_pattern: access_pattern.clone(),
            parallelization_potential: self.estimate_parallelization_potential(matrixshape),
            cache_efficiency: self.estimate_cache_efficiency(matrixshape, &access_pattern),
        })
    }

    /// Predict performance using machine learning model
    fn predict_performance(
        &self,
        operation: &str,
        workload: &WorkloadAnalysis,
        devices: &[GpuDeviceInfo],
    ) -> LinalgResult<Vec<PerformancePrediction>> {
        let predictor = self.performance_predictor.lock().map_err(|_| {
            LinalgError::ComputationError("Failed to lock performance predictor".to_string())
        })?;

        let mut predictions = Vec::new();

        // CPU prediction
        let cpu_prediction = predictor.predict_cpu_performance(operation, workload)?;
        predictions.push(PerformancePrediction {
            device_type: "CPU".to_string(),
            estimated_time: cpu_prediction.estimated_time,
            estimated_energy: cpu_prediction.estimated_energy,
            estimated_memory: cpu_prediction.estimated_memory,
            confidence_score: cpu_prediction.confidence_score,
        });

        // GPU predictions for each available device
        for (idx, device) in devices.iter().enumerate() {
            let gpu_prediction = predictor.predict_gpu_performance(operation, workload, device)?;
            predictions.push(PerformancePrediction {
                device_type: format!("GPU_{}", idx),
                estimated_time: gpu_prediction.estimated_time,
                estimated_energy: gpu_prediction.estimated_energy,
                estimated_memory: gpu_prediction.estimated_memory,
                confidence_score: gpu_prediction.confidence_score,
            });
        }

        Ok(predictions)
    }

    /// Multi-objective optimization considering time, energy, and memory
    fn optimize_multi_objective(
        &self,
        predictions: &[PerformancePrediction],
    ) -> LinalgResult<OptimalChoice> {
        if predictions.is_empty() {
            return Err(LinalgError::ComputationError(
                "No performance predictions available".to_string(),
            ));
        }

        // Weights for different objectives (could be made configurable)
        let time_weight = 0.5;
        let energy_weight = 0.3;
        let memory_weight = 0.2;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_choice = &predictions[0];

        for prediction in predictions {
            // Normalize metrics (lower is better for all)
            let time_score = 1.0 / (1.0 + prediction.estimated_time);
            let energy_score = 1.0 / (1.0 + prediction.estimated_energy);
            let memory_score = 1.0 / (1.0 + prediction.estimated_memory as f64);

            // Weighted score with confidence factor
            let total_score = (time_weight * time_score
                + energy_weight * energy_score
                + memory_weight * memory_score)
                * prediction.confidence_score;

            if total_score > best_score {
                best_score = total_score;
                best_choice = prediction;
            }
        }

        Ok(OptimalChoice {
            selected_device: best_choice.device_type.clone(),
            expected_performance: best_choice.clone(),
            optimization_score: best_score,
            reasoning: self.generate_reasoning(best_choice, predictions),
        })
    }

    /// Apply adaptive threshold learning to final decision
    fn apply_adaptive_thresholds(
        &self,
        operation: &str,
        optimal_choice: &OptimalChoice,
    ) -> LinalgResult<DispatchDecision> {
        let use_gpu = optimal_choice.selected_device != "CPU";

        Ok(DispatchDecision {
            use_gpu,
            selected_device: optimal_choice.selected_device.clone(),
            reasoning: optimal_choice.reasoning.clone(),
            confidence: optimal_choice.expected_performance.confidence_score,
            estimated_performance: optimal_choice.expected_performance.clone(),
        })
    }

    /// Generate human-readable reasoning for the dispatch decision
    fn generate_reasoning(
        &self,
        selected: &PerformancePrediction,
        all_options: &[PerformancePrediction],
    ) -> String {
        let cpu_option = all_options.iter().find(|p| p.device_type == "CPU");

        match cpu_option {
            Some(_cpu) if selected.device_type == "CPU" => {
                format!(
                    "Selected CPU: {:.3}s execution time vs GPU alternatives. \
                     Lower overhead and better cache efficiency for this workload.",
                    selected.estimated_time
                )
            }
            Some(cpu) => {
                let speedup = cpu.estimated_time / selected.estimated_time;
                format!(
                    "Selected {}: {:.2}x speedup over CPU ({:.3}s vs {:.3}s). \
                     High compute intensity justifies GPU acceleration.",
                    selected.device_type, speedup, selected.estimated_time, cpu.estimated_time
                )
            }
            None => {
                format!(
                    "Selected {} with {:.3}s estimated execution time.",
                    selected.device_type, selected.estimated_time
                )
            }
        }
    }

    // Helper methods for workload analysis
    fn estimate_compute_intensity(&self, operation: &str, shape: (usize, usize)) -> f64 {
        match operation {
            "matmul" => (shape.0 * shape.1 * shape.1) as f64 / (shape.0 * shape.1 * 2) as f64,
            "matvec" => (shape.0 * shape.1 * 2) as f64 / (shape.0 + shape.1) as f64,
            "norm" => 2.0, // 2 operations per element
            _ => 1.0,      // Default compute intensity
        }
    }

    fn estimate_memory_requirements(&self, shape: (usize, usize)) -> usize {
        let elements = shape.0 * shape.1;
        elements * std::mem::size_of::<T>()
    }

    fn detect_access_pattern(&self, operation: &str, shape: (usize, usize)) -> MemoryAccessPattern {
        match operation {
            "matmul" => {
                if shape.0 > 1024 && shape.1 > 1024 {
                    MemoryAccessPattern::Blocked(64, 64) // Typical block size for large matrices
                } else {
                    MemoryAccessPattern::Sequential
                }
            }
            "matvec" => MemoryAccessPattern::Sequential,
            "transpose" => MemoryAccessPattern::Strided(shape.1),
            _ => MemoryAccessPattern::Sequential,
        }
    }

    fn estimate_parallelization_potential(&self, shape: (usize, usize)) -> f64 {
        // Simple heuristic based on problem size
        let total_elements = shape.0 * shape.1;
        if total_elements > 1_000_000 {
            0.9 // High parallelization potential for large problems
        } else if total_elements > 10_000 {
            0.6 // Medium parallelization potential
        } else {
            0.2 // Low parallelization potential for small problems
        }
    }

    fn estimate_cache_efficiency(
        &self,
        shape: (usize, usize),
        pattern: &MemoryAccessPattern,
    ) -> f64 {
        match pattern {
            MemoryAccessPattern::Sequential => 0.9,
            MemoryAccessPattern::Blocked(_, _) => 0.8,
            MemoryAccessPattern::Strided(stride) => {
                if *stride < 8 {
                    0.7
                } else {
                    0.3
                }
            }
            MemoryAccessPattern::Random => 0.2,
            MemoryAccessPattern::Hierarchical => 0.6,
        }
    }
}

// Implementation stubs for the supporting structures
impl GpuPerformancePredictor {
    pub fn new() -> Self {
        Self {
            historical_data: HashMap::new(),
            model_coefficients: HashMap::new(),
            confidence_scores: HashMap::new(),
        }
    }

    pub fn predict_cpu_performance(
        &self,
        operation: &str,
        _workload: &WorkloadAnalysis,
    ) -> LinalgResult<PerformancePrediction> {
        // Simplified prediction - in practice would use sophisticated ML models
        Ok(PerformancePrediction {
            device_type: "CPU".to_string(),
            estimated_time: 0.1,
            estimated_energy: 10.0,
            estimated_memory: 1024,
            confidence_score: 0.8,
        })
    }

    pub fn predict_gpu_performance(
        &self,
        operation: &str,
        _workload: &WorkloadAnalysis,
        device: &GpuDeviceInfo,
    ) -> LinalgResult<PerformancePrediction> {
        // Simplified prediction - in practice would use sophisticated ML models
        Ok(PerformancePrediction {
            device_type: "GPU".to_string(),
            estimated_time: 0.05,
            estimated_energy: 25.0,
            estimated_memory: 2048,
            confidence_score: 0.7,
        })
    }
}

impl WorkloadAnalyzer {
    pub fn new() -> Self {
        Self {
            sparsity_cache: HashMap::new(),
            access_patterns: HashMap::new(),
            compute_intensity: HashMap::new(),
        }
    }
}

impl AdaptiveThresholdOptimizer {
    pub fn new() -> Self {
        Self {
            current_thresholds: HashMap::new(),
            learning_rate: 0.01,
            threshold_performance: HashMap::new(),
        }
    }

    pub fn update_threshold_performance(
        &mut self,
        operation: &str,
        performance: f64,
        used_gpu: bool,
    ) {
        let history = self
            .threshold_performance
            .entry(operation.to_string())
            .or_insert_with(VecDeque::new);

        // Keep only recent history
        if history.len() >= 100 {
            history.pop_front();
        }

        let current_threshold = self
            .current_thresholds
            .get(operation)
            .copied()
            .unwrap_or(50000);
        history.push_back((current_threshold, performance, used_gpu));

        // Simple threshold adaptation logic
        if history.len() >= 10 {
            let avg_performance =
                history.iter().map(|(_, p, _)| p).sum::<f64>() / history.len() as f64;
            let gpu_usage_rate =
                history.iter().filter(|(_, _, gpu)| *gpu).count() as f64 / history.len() as f64;

            // Adjust threshold based on performance and GPU usage
            let threshold_adjustment = if gpu_usage_rate > 0.8 && avg_performance > 0.5 {
                -1000 // Lower threshold to use GPU more
            } else if gpu_usage_rate < 0.2 {
                1000 // Raise threshold to use CPU more
            } else {
                0
            };

            if threshold_adjustment != 0 {
                let new_threshold =
                    (current_threshold as i32 + threshold_adjustment).max(1000) as usize;
                self.current_thresholds
                    .insert(operation.to_string(), new_threshold);
            }
        }
    }
}
