//! FFT Planning Module
//!
//! This module provides an advanced planning system for FFT operations, combining
//! caching, serialization, and auto-tuning to optimize performance. It builds on the
//! functionality provided by the plan_cache and plan_serialization modules.
//!
//! Key features:
//! - Unified planning interface for different FFT backends
//! - Multi-dimensional FFT planning
//! - Adaptive plan selection based on input characteristics
//! - Integration with auto-tuning for hardware-specific optimizations
//! - Support for both runtime and ahead-of-time planning
//! - Plan pruning and management to optimize memory usage

use crate::auto_tuning::{AutoTuneConfig, AutoTuner, FftVariant};
use crate::backend::BackendContext;
use crate::error::{FFTError, FFTResult};
use crate::plan_serialization::{PlanMetrics, PlanSerializationManager};

use ndarray::{ArrayBase, Data, Dimension};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Enum for different planning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlanningStrategy {
    /// Always create a new plan
    AlwaysNew,
    /// Try cache first, create new plan if not found
    #[default]
    CacheFirst,
    /// Try serialized plans first, then cache, then create new
    SerializedFirst,
    /// Auto-tune to find the best plan for the current hardware
    AutoTuned,
}

/// Configuration options for FFT planning
#[derive(Debug, Clone)]
pub struct PlanningConfig {
    /// Planning strategy to use
    pub strategy: PlanningStrategy,
    /// Whether to measure plan performance
    pub measure_performance: bool,
    /// Path to serialized plans database
    pub serialized_db_path: Option<String>,
    /// Auto-tuning configuration (if AutoTuned strategy is selected)
    pub auto_tune_config: Option<AutoTuneConfig>,
    /// Maximum number of plans to keep in memory
    pub max_cached_plans: usize,
    /// Maximum age for cached plans
    pub max_plan_age: Duration,
    /// Whether to use parallel execution for planning
    pub parallel_planning: bool,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            strategy: PlanningStrategy::default(),
            measure_performance: true,
            serialized_db_path: None,
            auto_tune_config: None,
            max_cached_plans: 128,
            max_plan_age: Duration::from_secs(3600), // 1 hour
            parallel_planning: true,
        }
    }
}

/// Multidimensional FFT plan that handles different array shapes and layouts
#[derive(Clone)]
pub struct FftPlan {
    /// Array shape the plan is optimized for
    shape: Vec<usize>,
    /// Whether the plan is for a forward or inverse transform
    /// Kept for future compatibility
    #[allow(dead_code)]
    forward: bool,
    /// Backend-specific internal plan representation
    internal_plan: Arc<dyn rustfft::Fft<f64>>,
    /// Performance metrics for this plan
    metrics: Option<PlanMetrics>,
    /// Backend used to create this plan
    /// Preserved for potential switching between backends
    #[allow(dead_code)]
    backend: PlannerBackend,
    /// Auto-tuning information (if applicable)
    auto_tune_info: Option<FftVariant>,
    /// Last time this plan was used
    last_used: Instant,
    /// Number of times this plan has been used
    usage_count: usize,
}

impl FftPlan {
    /// Create a new FFT plan for the given shape and direction
    pub fn new(
        shape: &[usize],
        forward: bool,
        planner: &mut rustfft::FftPlanner<f64>,
        backend: PlannerBackend,
    ) -> Self {
        // For multidimensional transforms, we'll use the flattened size
        let size = shape.iter().product();

        let internal_plan = if forward {
            planner.plan_fft_forward(size)
        } else {
            planner.plan_fft_inverse(size)
        };

        Self {
            shape: shape.to_vec(),
            forward,
            internal_plan,
            metrics: None,
            backend,
            auto_tune_info: None,
            last_used: Instant::now(),
            usage_count: 0,
        }
    }

    /// Get the internal rustfft plan
    pub fn get_internal(&self) -> Arc<dyn rustfft::Fft<f64>> {
        self.internal_plan.clone()
    }

    /// Record usage of this plan
    pub fn record_usage(&mut self) {
        self.usage_count += 1;
        self.last_used = Instant::now();
    }

    /// Get shape this plan is optimized for
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Check if this plan is compatible with the given shape
    pub fn is_compatible_with(&self, shape: &[usize]) -> bool {
        if self.shape.len() != shape.len() {
            return false;
        }

        self.shape.iter().zip(shape.iter()).all(|(&a, &b)| a == b)
    }

    /// Get plan metrics
    pub fn metrics(&self) -> Option<&PlanMetrics> {
        self.metrics.as_ref()
    }

    /// Set plan metrics
    pub fn set_metrics(&mut self, metrics: PlanMetrics) {
        self.metrics = Some(metrics);
    }
}

/// Planner for FFT operations that combines caching, serialization and auto-tuning
pub struct AdvancedFftPlanner {
    /// Configuration options
    config: PlanningConfig,
    /// In-memory plan cache
    cache: Arc<Mutex<HashMap<PlanKey, FftPlan>>>,
    /// Plan serialization manager (if enabled)
    serialization_manager: Option<PlanSerializationManager>,
    /// Auto-tuner (if enabled)
    auto_tuner: Option<AutoTuner>,
    /// Internal rustfft planner
    internal_planner: rustfft::FftPlanner<f64>,
}

/// Backend type for FFT operations
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum PlannerBackend {
    /// Default rustfft backend
    #[default]
    RustFFT,
    /// FFTW-compatible backend
    FFTW,
    /// CUDA-accelerated backend
    CUDA,
    /// Custom backend implementation
    Custom(String),
}

/// Key for plan cache lookup
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct PlanKey {
    /// Shape of the FFT
    shape: Vec<usize>,
    /// Direction of the transform
    forward: bool,
    /// Backend identifier
    backend: PlannerBackend,
}

/// Advanced FFT planner for managing plans
impl AdvancedFftPlanner {
    /// Create a new FftPlanner with default configuration
    pub fn new() -> Self {
        Self::with_config(PlanningConfig::default())
    }

    /// Create a new FftPlanner with custom configuration
    pub fn with_config(config: PlanningConfig) -> Self {
        let serialization_manager = config
            .serialized_db_path
            .as_ref()
            .map(PlanSerializationManager::new);

        let auto_tuner = if config.strategy == PlanningStrategy::AutoTuned {
            let tuner = AutoTuner::new();
            if let Some(_auto_config) = &config.auto_tune_config {
                // Configure the auto-tuner here if needed
                // This is a simplified implementation
            }
            Some(tuner)
        } else {
            None
        };

        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            serialization_manager,
            auto_tuner,
            internal_planner: rustfft::FftPlanner::new(),
        }
    }

    /// Clear the plan cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Get or create a plan for the given shape and direction
    pub fn plan_fft(
        &mut self,
        shape: &[usize],
        forward: bool,
        backend: PlannerBackend,
    ) -> FFTResult<Arc<FftPlan>> {
        let key = PlanKey {
            shape: shape.to_vec(),
            forward,
            backend: backend.clone(),
        };

        // Check in-memory cache first if strategy allows
        if self.config.strategy == PlanningStrategy::CacheFirst
            || self.config.strategy == PlanningStrategy::SerializedFirst
        {
            if let Ok(mut cache) = self.cache.lock() {
                if let Some(plan) = cache.get_mut(&key) {
                    plan.record_usage();
                    return Ok(Arc::new(plan.clone()));
                }
            }
        }

        // Check serialized plans if strategy is SerializedFirst
        if self.config.strategy == PlanningStrategy::SerializedFirst {
            if let Some(manager) = &self.serialization_manager {
                // For simplicity, we'll use the primary dimension size for serialization
                // In a full implementation, this would handle multidimensional plans better
                let size = shape.iter().product();

                if manager.plan_exists(size, forward) {
                    if let Some((_plan_info, metrics)) =
                        manager.get_best_plan_metrics(size, forward)
                    {
                        // Create a new plan using the cached metadata
                        let mut plan = self.create_new_plan(shape, forward, backend.clone())?;
                        plan.set_metrics(metrics.clone());

                        // Store in cache
                        if let Ok(mut cache) = self.cache.lock() {
                            cache.insert(key, plan.clone());
                        }

                        return Ok(Arc::new(plan));
                    }
                }
            }
        }

        // Use auto-tuning if strategy is AutoTuned
        if self.config.strategy == PlanningStrategy::AutoTuned {
            if let Some(tuner) = &self.auto_tuner {
                // This is a simplified example - a full implementation would do much more
                let size = shape.iter().product();
                let variant = tuner.get_best_variant(size, forward);

                // In a real implementation, we would use the variant to create an optimized plan
                let mut plan = self.create_new_plan(shape, forward, backend)?;
                plan.auto_tune_info = Some(variant);

                // Store in cache
                if let Ok(mut cache) = self.cache.lock() {
                    cache.insert(key, plan.clone());
                }

                return Ok(Arc::new(plan));
            }
        }

        // Default: create a new plan
        let plan = self.create_new_plan(shape, forward, backend)?;

        // Store in cache if caching is enabled
        if self.config.strategy != PlanningStrategy::AlwaysNew {
            if let Ok(mut cache) = self.cache.lock() {
                // Clean up old entries if we're at capacity
                if cache.len() >= self.config.max_cached_plans {
                    self.evict_old_entries(&mut cache);
                }

                cache.insert(key, plan.clone());
            }
        }

        Ok(Arc::new(plan))
    }

    /// Create a new plan without using the cache
    fn create_new_plan(
        &mut self,
        shape: &[usize],
        forward: bool,
        backend: PlannerBackend,
    ) -> FFTResult<FftPlan> {
        // Measure plan creation time if enabled
        let start = Instant::now();

        let plan = FftPlan::new(shape, forward, &mut self.internal_planner, backend);

        let elapsed = start.elapsed();

        // Record metrics for the new plan if measurement is enabled
        if self.config.measure_performance {
            if let Some(manager) = &self.serialization_manager {
                // For simplicity, we'll use the primary dimension size
                // In a full implementation, this would handle multidimensional plans better
                let size = shape.iter().product();

                let plan_info = manager.create_plan_info(size, forward);
                let _ = manager.record_plan_usage(&plan_info, elapsed.as_nanos() as u64);
            }
        }

        Ok(plan)
    }

    /// Evict old entries from the cache (LRU-style)
    fn evict_old_entries(&self, cache: &mut HashMap<PlanKey, FftPlan>) {
        // Remove entries older than max_age
        let max_age = self.config.max_plan_age;
        cache.retain(|_, v| v.last_used.elapsed() <= max_age);

        // If still over capacity, remove least recently used
        let max_entries = self.config.max_cached_plans;
        while cache.len() >= max_entries {
            if let Some((key_to_remove, _)) = cache
                .iter()
                .min_by_key(|(_, v)| (v.last_used, v.usage_count))
                .map(|(k, _)| (k.clone(), ()))
            {
                cache.remove(&key_to_remove);
            } else {
                break;
            }
        }
    }

    /// Plan a 1D FFT
    pub fn plan_fft_1d<S, D>(
        &mut self,
        arr: &ArrayBase<S, D>,
        forward: bool,
    ) -> FFTResult<Arc<FftPlan>>
    where
        S: Data<Elem = Complex64>,
        D: Dimension,
    {
        let shape = arr.shape().to_vec();
        self.plan_fft(&shape, forward, PlannerBackend::default())
    }

    /// Plan a 2D FFT
    pub fn plan_fft_2d<S, D>(
        &mut self,
        arr: &ArrayBase<S, D>,
        forward: bool,
    ) -> FFTResult<Arc<FftPlan>>
    where
        S: Data<Elem = Complex64>,
        D: Dimension,
    {
        if arr.ndim() != 2 {
            return Err(FFTError::ValueError(
                "Input array must be 2-dimensional".to_string(),
            ));
        }

        let shape = arr.shape().to_vec();
        self.plan_fft(&shape, forward, PlannerBackend::default())
    }

    /// Plan an N-dimensional FFT
    pub fn plan_fft_nd<S, D>(
        &mut self,
        arr: &ArrayBase<S, D>,
        forward: bool,
    ) -> FFTResult<Arc<FftPlan>>
    where
        S: Data<Elem = Complex64>,
        D: Dimension,
    {
        let shape = arr.shape().to_vec();
        self.plan_fft(&shape, forward, PlannerBackend::default())
    }

    /// Pre-compute plans for common sizes
    pub fn precompute_common_sizes(&mut self, sizes: &[&[usize]]) -> FFTResult<()> {
        for &shape in sizes {
            // Create both forward and inverse plans
            let _ = self.plan_fft(shape, true, PlannerBackend::default())?;
            let _ = self.plan_fft(shape, false, PlannerBackend::default())?;
        }

        Ok(())
    }

    /// Save all plans to disk
    pub fn save_plans(&self) -> FFTResult<()> {
        if let Some(manager) = &self.serialization_manager {
            manager.save_database()?;
        }

        Ok(())
    }
}

impl Default for AdvancedFftPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Global FFT planner instance
static GLOBAL_FFT_PLANNER: std::sync::OnceLock<Mutex<AdvancedFftPlanner>> =
    std::sync::OnceLock::new();

/// Get the global FFT planner instance
pub fn get_global_planner() -> &'static Mutex<AdvancedFftPlanner> {
    GLOBAL_FFT_PLANNER.get_or_init(|| Mutex::new(AdvancedFftPlanner::new()))
}

/// Initialize the global FFT planner with custom configuration
pub fn init_global_planner(config: PlanningConfig) -> Result<(), &'static str> {
    GLOBAL_FFT_PLANNER
        .set(Mutex::new(AdvancedFftPlanner::with_config(config)))
        .map_err(|_| "Global FFT planner already initialized")
}

/// Plan wrapper that provides context and execution methods
pub struct FftPlanExecutor {
    /// The underlying FFT plan
    plan: Arc<FftPlan>,
    /// Execution context for this plan
    /// Reserved for future optimizations
    #[allow(dead_code)]
    context: Option<BackendContext>,
}

impl FftPlanExecutor {
    /// Create a new executor for the given plan
    pub fn new(plan: Arc<FftPlan>) -> Self {
        Self {
            plan,
            context: None,
        }
    }

    /// Create a new executor with a specific context
    pub fn with_context(plan: Arc<FftPlan>, context: BackendContext) -> Self {
        Self {
            plan,
            context: Some(context),
        }
    }

    /// Execute the plan on the given input/output buffers
    pub fn execute(&self, input: &[Complex64], output: &mut [Complex64]) -> FFTResult<()> {
        // In a real implementation, we would use the context and backend
        // to select the appropriate execution strategy

        // For now, just use the rustfft plan directly
        let internal_plan = self.plan.get_internal();

        // Ensure buffer sizes match expected size
        let expected_size: usize = self.plan.shape().iter().product();
        if input.len() != expected_size || output.len() != expected_size {
            return Err(FFTError::ValueError(format!(
                "Buffer size mismatch: expected {}, got input={}, output={}",
                expected_size,
                input.len(),
                output.len()
            )));
        }

        // Execute the transform
        // rustfft requires in-place execution, so copy input to output first
        output.copy_from_slice(input);

        // Then execute in-place
        let mut scratch = vec![Complex64::default(); internal_plan.get_inplace_scratch_len()];
        internal_plan.process_with_scratch(output, &mut scratch);

        Ok(())
    }

    /// Execute the plan in-place on the given buffer
    pub fn execute_inplace(&self, buffer: &mut [Complex64]) -> FFTResult<()> {
        // Similar to execute, but in-place
        let internal_plan = self.plan.get_internal();

        // Ensure buffer size matches expected size
        let expected_size: usize = self.plan.shape().iter().product();
        if buffer.len() != expected_size {
            return Err(FFTError::ValueError(format!(
                "Buffer size mismatch: expected {}, got {}",
                expected_size,
                buffer.len()
            )));
        }

        // Execute the transform in-place
        let mut scratch = vec![Complex64::default(); internal_plan.get_inplace_scratch_len()];
        internal_plan.process_with_scratch(buffer, &mut scratch);

        Ok(())
    }

    /// Get the plan this executor uses
    pub fn plan(&self) -> &FftPlan {
        &self.plan
    }
}

/// Builder for creating customized FFT plans
pub struct PlanBuilder {
    /// Planning configuration
    config: PlanningConfig,
    /// Shape for the transform
    shape: Option<Vec<usize>>,
    /// Direction of the transform
    forward: bool,
    /// Backend to use
    backend: PlannerBackend,
}

impl PlanBuilder {
    /// Create a new plan builder
    pub fn new() -> Self {
        Self {
            config: PlanningConfig::default(),
            shape: None,
            forward: true,
            backend: PlannerBackend::default(),
        }
    }

    /// Set the shape for the plan
    pub fn shape(mut self, shape: &[usize]) -> Self {
        self.shape = Some(shape.to_vec());
        self
    }

    /// Set the direction for the plan
    pub fn forward(mut self, forward: bool) -> Self {
        self.forward = forward;
        self
    }

    /// Set the backend for the plan
    pub fn backend(mut self, backend: PlannerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the planning strategy
    pub fn strategy(mut self, strategy: PlanningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Enable or disable performance measurement
    pub fn measure_performance(mut self, enable: bool) -> Self {
        self.config.measure_performance = enable;
        self
    }

    /// Set the path for serialized plans
    pub fn serialized_db_path(mut self, path: &str) -> Self {
        self.config.serialized_db_path = Some(path.to_string());
        self
    }

    /// Set auto-tuning configuration
    pub fn auto_tune_config(mut self, config: AutoTuneConfig) -> Self {
        self.config.auto_tune_config = Some(config);
        self
    }

    /// Set maximum number of cached plans
    pub fn max_cached_plans(mut self, max: usize) -> Self {
        self.config.max_cached_plans = max;
        self
    }

    /// Set maximum age for cached plans
    pub fn max_plan_age(mut self, age: Duration) -> Self {
        self.config.max_plan_age = age;
        self
    }

    /// Enable or disable parallel planning
    pub fn parallel_planning(mut self, enable: bool) -> Self {
        self.config.parallel_planning = enable;
        self
    }

    /// Build the plan
    pub fn build(self) -> FFTResult<Arc<FftPlan>> {
        let shape = self
            .shape
            .ok_or_else(|| FFTError::ValueError("Cannot build plan without shape".to_string()))?;

        let mut planner = AdvancedFftPlanner::with_config(self.config);
        planner.plan_fft(&shape, self.forward, self.backend)
    }
}

impl Default for PlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Plan ahead-of-time for common FFT sizes
///
/// This function pre-computes plans for commonly used FFT sizes
/// and stores them in the cache for faster initialization later.
///
/// # Arguments
///
/// * `sizes` - List of common FFT sizes to pre-compute plans for
/// * `db_path` - Optional path to store serialized plans
///
/// # Returns
///
/// Result indicating success or failure
pub fn plan_ahead_of_time(sizes: &[usize], db_path: Option<&str>) -> FFTResult<()> {
    let mut config = PlanningConfig::default();
    if let Some(path) = db_path {
        config.serialized_db_path = Some(path.to_string());
        config.strategy = PlanningStrategy::SerializedFirst;
    }

    let mut planner = AdvancedFftPlanner::with_config(config);

    // Convert to shapes (assuming 1D transforms for simplicity)
    let shapes: Vec<Vec<usize>> = sizes.iter().map(|&s| vec![s]).collect();

    for shape in shapes {
        // Create both forward and inverse plans
        let _ = planner.plan_fft(&shape, true, PlannerBackend::default())?;
        let _ = planner.plan_fft(&shape, false, PlannerBackend::default())?;
    }

    // Save plans if serialization is enabled
    planner.save_plans()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tempfile::tempdir;

    #[test]
    fn test_plan_basic() {
        let mut planner = AdvancedFftPlanner::new();
        let shape = vec![8, 8];

        // Create a plan
        let plan = planner
            .plan_fft(&shape, true, PlannerBackend::default())
            .unwrap();

        // Check that the plan has the right shape
        assert_eq!(plan.shape(), &shape);
        assert!(plan.is_compatible_with(&shape));

        // Check that a different shape is not compatible
        assert!(!plan.is_compatible_with(&[16, 16]));
    }

    #[test]
    fn test_plan_executor() {
        let mut planner = AdvancedFftPlanner::new();
        let shape = vec![8];

        // Create a plan
        let plan = planner
            .plan_fft(&shape, true, PlannerBackend::default())
            .unwrap();

        // Create an executor
        let executor = FftPlanExecutor::new(plan);

        // Create some test data
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut output = vec![Complex64::default(); 8];

        // Execute the plan
        executor.execute(&input, &mut output).unwrap();

        // Check that the output makes sense for this input
        // (an impulse at the beginning should have a flat frequency response)
        for val in &output {
            // The magnitude should be approximately the same for all frequencies
            let magnitude = (val.re.powi(2) + val.im.powi(2)).sqrt();
            assert!((magnitude - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_plan_builder() {
        let builder = PlanBuilder::new()
            .shape(&[16])
            .forward(true)
            .strategy(PlanningStrategy::AlwaysNew)
            .measure_performance(true);

        let plan = builder.build().unwrap();

        assert_eq!(plan.shape(), &[16]);
    }

    #[test]
    fn test_serialization() {
        // Create a temporary directory for test
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_plan_db.json");

        // Create a planner with serialization enabled
        let mut config = PlanningConfig::default();
        config.serialized_db_path = Some(db_path.to_str().unwrap().to_string());
        config.strategy = PlanningStrategy::SerializedFirst;

        let mut planner = AdvancedFftPlanner::with_config(config);

        // Create a plan
        let shape = vec![32];
        let _ = planner
            .plan_fft(&shape, true, PlannerBackend::default())
            .unwrap();

        // Save plans
        planner.save_plans().unwrap();

        // Check that the file exists
        assert!(db_path.exists());
    }

    #[test]
    fn test_global_planner() {
        // Get the global planner
        let planner = get_global_planner();

        // Create a plan with the global planner
        let mut planner_guard = planner.lock().unwrap();
        let shape = vec![64];
        let plan = planner_guard
            .plan_fft(&shape, true, PlannerBackend::default())
            .unwrap();

        assert_eq!(plan.shape(), &shape);
    }

    #[test]
    fn test_ahead_of_time_planning() {
        // Create a temporary directory for test
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("ahead_of_time.json");

        // Plan ahead of time for some common sizes
        let sizes = [8, 16, 32, 64];
        plan_ahead_of_time(&sizes, Some(db_path.to_str().unwrap())).unwrap();

        // Check that the file exists
        assert!(db_path.exists());
    }
}
