//! Comprehensive performance integration for neural networks
//!
//! This module provides a unified interface for all performance optimizations available
//! in scirs2-neural, including:
//! - CPU optimizations (SIMD, threading, memory efficiency)
//! - GPU acceleration
//! - TPU support
//! - JIT compilation
//! - Automatic optimization selection
//! - Performance monitoring and profiling

use crate::error::{NeuralError, Result};
use crate::jit::{JITCompiler, JITOperation};
use crate::performance::{PerformanceOptimizer, PerformanceStats};
use crate::tpu::{TPUOperation, TPURuntime};
use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::Div;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
/// Unified performance manager for all acceleration methods
pub struct UnifiedPerformanceManager {
    /// CPU performance optimizer
    cpu_optimizer: PerformanceOptimizer,
    /// JIT compiler for dynamic optimization
    jit_compiler: Option<JITCompiler>,
    /// TPU runtime for tensor processing units
    tpu_runtime: Option<TPURuntime>,
    /// GPU acceleration status
    gpu_available: bool,
    /// Performance monitoring
    monitor: Arc<RwLock<PerformanceMonitor>>,
    /// Automatic optimization strategy
    auto_optimization: AutoOptimizationStrategy,
    /// Operation cache for optimization decisions
    operation_cache: Arc<RwLock<HashMap<OperationKey, OptimizationChoice>>>,
}
/// Performance monitoring and analytics
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Execution times for different optimization strategies
    execution_times: HashMap<OptimizationChoice, Vec<Duration>>,
    /// Memory usage statistics
    memory_usage: HashMap<OptimizationChoice, Vec<usize>>,
    /// Success/failure rates
    success_rates: HashMap<OptimizationChoice, (u64, u64)>, // (successes, total)
    /// Device utilization
    device_utilization: HashMap<String, f64>,
    /// Performance trends over time
    performance_trends: Vec<PerformanceSample>,
    /// Total operations executed
    total_operations: u64,
/// Single performance measurement sample
pub struct PerformanceSample {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Operation type
    pub operation: String,
    /// Optimization strategy used
    pub strategy: OptimizationChoice,
    /// Execution time
    pub execution_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Success flag
    pub success: bool,
/// Available optimization strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationChoice {
    /// CPU SIMD acceleration
    CPUSimd,
    /// CPU parallel processing
    CPUParallel,
    /// CPU serial (baseline)
    CPUSerial,
    /// JIT compiled operations
    JIT,
    /// TPU acceleration
    TPU,
    /// GPU acceleration (placeholder for future)
    GPU,
    /// Hybrid approach combining multiple strategies
    Hybrid(Vec<OptimizationChoice>),
/// Automatic optimization strategy
pub enum AutoOptimizationStrategy {
    /// Always use the fastest available method
    AlwaysFastest,
    /// Use most energy efficient method
    EnergyEfficient,
    /// Balance speed and energy consumption
    Balanced,
    /// Adaptive learning based on historical performance
    Adaptive {
        /// Learning rate for adaptation
        learning_rate: f64,
        /// Window size for performance history
        window_size: usize,
    },
    /// Custom strategy with user-defined rules
    Custom(Box<dyn Fn(&OperationContext) -> OptimizationChoice + Send + Sync>),
/// Context information for optimization decisions
pub struct OperationContext {
    /// Type of operation
    pub operation_type: String,
    /// Input tensor shapes
    pub inputshapes: Vec<Vec<usize>>,
    /// Expected output shapes
    pub outputshapes: Vec<Vec<usize>>,
    /// Memory constraints (bytes)
    pub memory_limit: Option<usize>,
    /// Time constraints (milliseconds)
    pub time_limit: Option<u64>,
    /// Energy constraints (relative scale 0-1)
    pub energy_limit: Option<f64>,
    /// Batch size
    pub batch_size: usize,
/// Key for caching optimization decisions
pub struct OperationKey {
    /// Operation type identifier
    /// Input shape signature
    pub shape_signature: String,
    /// Parameter signature (for operations with parameters)
    pub param_signature: String,
/// Comprehensive performance statistics
pub struct UnifiedPerformanceStats {
    /// CPU performance statistics
    pub cpu_stats: PerformanceStats,
    /// JIT compilation statistics
    pub jit_stats: Option<JITStats>,
    /// TPU statistics
    pub tpu_stats: Option<TPUStats>,
    /// GPU statistics (placeholder)
    pub gpu_stats: Option<GPUStats>,
    /// Cross-platform statistics
    pub unified_stats: UnifiedStats,
/// JIT compilation statistics
pub struct JITStats {
    /// Number of kernels compiled
    pub kernels_compiled: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average compilation time
    pub avg_compile_time_ms: f64,
    /// Total execution time
    pub total_execution_time_ms: f64,
/// TPU execution statistics
pub struct TPUStats {
    /// Number of operations executed
    pub operations_executed: u64,
    /// Average execution time
    pub avg_execution_time_ms: f64,
    pub avg_utilization: f64,
    /// Memory usage
    pub peak_memory_usage: usize,
/// GPU statistics (placeholder for future implementation)
pub struct GPUStats {
    /// GPU device information
    pub device_info: String,
    /// Utilization percentage
    pub utilization: f64,
/// Cross-platform unified statistics
pub struct UnifiedStats {
    /// Total operations across all devices
    pub total_operations: u64,
    /// Performance improvement over baseline
    pub avg_speedup: f64,
    /// Energy efficiency improvement
    pub energy_efficiency: f64,
    /// Most used optimization strategy
    pub preferred_strategy: OptimizationChoice,
    /// Strategy distribution
    pub strategy_distribution: HashMap<OptimizationChoice, f64>,
impl UnifiedPerformanceManager {
    /// Create a new unified performance manager
    pub fn new() -> Result<Self> {
        // Initialize CPU optimizer
        let cpu_optimizer = PerformanceOptimizer::new(
            Some(512),  // chunk_size
            Some(2048), // max_memory_mb
            None,       // auto-detect threads
            true,       // enable_profiling
        )?;
        // Try to initialize JIT compiler
        let jit_compiler = match JITCompiler::new(JITCompiler::detect_target_architecture()) {
            Ok(compiler) => Some(compiler),
            Err(_) => None, // JIT not available
        };
        // Try to initialize TPU runtime
        let tpu_runtime = match TPURuntime::initialize() {
            Ok(runtime) => Some(runtime),
            Err(_) => None, // TPU not available
        // Detect GPU availability (placeholder)
        let gpu_available = Self::detect_gpu_availability();
        Ok(Self {
            cpu_optimizer,
            jit_compiler,
            tpu_runtime,
            gpu_available,
            monitor: Arc::new(RwLock::new(PerformanceMonitor::new())),
            auto_optimization: AutoOptimizationStrategy::Adaptive {
                learning_rate: 0.1,
                window_size: 100,
            },
            operation_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    /// Execute an operation with automatic optimization selection
    pub fn execute_optimized<F: Float + Debug>(
        &mut self,
        operation_type: &str,
        inputs: &[&ArrayD<F>],
        context: OperationContext,
    ) -> Result<Vec<ArrayD<F>>> {
        let start_time = Instant::now();
        // Create operation key for caching
        let op_key = self.create_operation_key(operation_type, inputs, &context);
        // Select optimization strategy
        let strategy = self.select_optimization_strategy(&op_key, &context)?;
        // Execute with selected strategy
        let result = self.execute_with_strategy(&strategy, operation_type, inputs, &context);
        // Record performance metrics
        let execution_time = start_time.elapsed();
        let success = result.is_ok();
        let memory_usage = self.estimate_memory_usage(inputs);
        self.record_performance_sample(PerformanceSample {
            timestamp: start_time,
            operation: operation_type.to_string(),
            strategy: strategy.clone(),
            execution_time,
            memory_usage,
            success,
        });
        // Update optimization cache if successful
        if success {
            self.update_operation_cache(op_key, strategy);
        }
        result
    /// Execute matrix multiplication with best available optimization
    pub fn optimized_matmul(&mut self, a: &ArrayD<f32>, b: &ArrayD<f32>) -> Result<ArrayD<f32>> {
        let context = OperationContext {
            operation_type: "matmul".to_string(),
            inputshapes: vec![a.shape().to_vec(), b.shape().to_vec()],
            outputshapes: vec![vec![a.shape()[0], b.shape()[1]]],
            memory_limit: None,
            time_limit: None,
            energy_limit: None,
            batch_size: a.shape()[0],
        let results = self.execute_optimized("matmul", &[a, b], context)?;
        Ok(results.into_iter().next().unwrap())
    /// Execute convolution with best available optimization
    pub fn optimized_conv2d(
        input: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<ArrayD<f32>> {
        // Calculate output shape (simplified)
        let outputshape = self.calculate_conv2d_outputshape(input, kernel, stride, padding);
            operation_type: "conv2d".to_string(),
            inputshapes: vec![input.shape().to_vec(), kernel.shape().to_vec()],
            outputshapes: vec![outputshape],
            batch_size: input.shape()[0],
        // For now, delegate to CPU optimizer
        self.cpu_optimizer
            .optimized_conv2d(input, kernel, bias, stride, padding)
    /// Select the best optimization strategy for a given operation
    fn select_optimization_strategy(
        &self,
        op_key: &OperationKey,
        context: &OperationContext,
    ) -> Result<OptimizationChoice> {
        // Check cache first
        if let Some(cached_choice) = self.get_cached_optimization(op_key) {
            return Ok(cached_choice);
        match &self.auto_optimization {
            AutoOptimizationStrategy::AlwaysFastest => self.select_fastest_strategy(context),
            AutoOptimizationStrategy::EnergyEfficient => {
                self.select_energy_efficient_strategy(context)
            }
            AutoOptimizationStrategy::Balanced => self.select_balanced_strategy(context),
            AutoOptimizationStrategy::Adaptive {
                learning_rate: _,
                window_size,
            } => self.select_adaptive_strategy(context, *window_size),
            AutoOptimizationStrategy::Custom(selector) => Ok(selector(context)),
    /// Execute operation with specific strategy
    fn execute_with_strategy<F: Float + Debug>(
        strategy: &OptimizationChoice, context: &OperationContext,
        match strategy {
            OptimizationChoice::CPUSimd => {
                #[cfg(feature = "simd")]
                {
                    use crate::performance::SIMDOperations;
                    match operation_type {
                        "matmul" => {
                            if inputs.len() >= 2 {
                                let a_f32 = inputs[0].mapv(|x| x.to_f32().unwrap_or(0.0));
                                let b_f32 = inputs[1].mapv(|x| x.to_f32().unwrap_or(0.0));
                                let result =
                                    SIMDOperations::simd_matmul_f32(&a_f32.view(), &b_f32.view())?;
                                let result_f = result.mapv(|x| F::from(x).unwrap_or(F::zero()));
                                Ok(vec![result_f])
                            } else {
                                Err(NeuralError::InvalidArgument(
                                    "MatMul requires 2 inputs".to_string(),
                                ))
                            }
                        }
                        "elementwise_add" => {
                                    SIMDOperations::simd_add_f32(&a_f32.view(), &b_f32.view())?;
                                    "ElementwiseAdd requires 2 inputs".to_string(),
                        "relu" => {
                            if !inputs.is_empty() {
                                let input_f32 = inputs[0].mapv(|x| x.to_f32().unwrap_or(0.0));
                                let result = SIMDOperations::simd_relu_f32(&input_f32.view());
                                    "ReLU requires 1 input".to_string(),
                        "conv2d" => {
                                let kernel_f32 = inputs[1].mapv(|x| x.to_f32().unwrap_or(0.0));
                                // Use default stride and padding for now
                                let result = SIMDOperations::simd_conv2d_f32(
                                    &input_f32.view(),
                                    &kernel_f32.view(),
                                    None,
                                    (1, 1),
                                    (0, 0),
                                )?;
                                    "Conv2D requires 2 inputs".to_string(, _ => Err(NeuralError::NotImplemented(format!(
                            "Operation {} not implemented for CPU SIMD",
                            operation_type
                        ))),
                    }
                }
                #[cfg(not(feature = "simd"))]
                    Err(NeuralError::FeatureNotEnabled(
                        "SIMD feature not enabled".to_string(),
                    ))
            OptimizationChoice::CPUParallel => {
                match operation_type {
                    "matmul" => {
                        if inputs.len() >= 2 {
                            // Convert to f32 for CPU optimizer
                            let a_f32 = inputs[0].mapv(|x| x.to_f32().unwrap_or(0.0));
                            let b_f32 = inputs[1].mapv(|x| x.to_f32().unwrap_or(0.0));
                            let result = self.cpu_optimizer.optimized_matmul(&a_f32, &b_f32)?;
                            // Convert back to F
                            let result_f = result.mapv(|x| F::from(x).unwrap_or(F::zero()));
                            Ok(vec![result_f])
                        } else {
                            Err(NeuralError::InvalidArgument(
                                "MatMul requires 2 inputs".to_string(),
                            ))
                    "conv2d" => {
                            let input_f32 = inputs[0].mapv(|x| x.to_f32().unwrap_or(0.0));
                            let kernel_f32 = inputs[1].mapv(|x| x.to_f32().unwrap_or(0.0));
                            let result = self.cpu_optimizer.optimized_conv2d(
                                &input_f32,
                                &kernel_f32,
                                None,
                                (1, 1),
                                (0, 0),
                            )?;
                                "Conv2D requires 2 inputs".to_string(, _ => Err(NeuralError::NotImplemented(format!(
                        "Operation {} not implemented for CPU parallel",
                        operation_type
                    ))),
            OptimizationChoice::JIT => {
                if let Some(jit_compiler) = &self.jit_compiler {
                    let jit_op = self.convert_to_jit_operation(operation_type, inputs)?;
                    let outputshapes = self.infer_jit_outputshapes(&jit_op, inputs)?;
                    // Execute with JIT (simplified to first input conversion)
                    if !inputs.is_empty() {
                        let f32_input = inputs[0].mapv(|x| x.to_f32().unwrap_or(0.0));
                        let result = jit_compiler.compile_and_execute(
                            &jit_op,
                            &[&f32_input],
                            &outputshapes[0],
                        )?;
                        let result_f = result.mapv(|x| F::from(x).unwrap_or(F::zero()));
                        Ok(vec![result_f])
                    } else {
                        Err(NeuralError::InvalidArgument(
                            "No inputs provided".to_string(),
                        ))
                } else {
                    Err(NeuralError::DeviceError(
                        "JIT compiler not available".to_string(),
            OptimizationChoice::TPU => {
                if self.tpu_runtime.is_some() {
                    let tpu_op = self.convert_to_tpu_operation(operation_type)?;
                    let results = self.tpu_runtime.as_mut().unwrap().compile_and_execute(&tpu_op, inputs)?;
                    Ok(results
                        .into_iter()
                        .map(|arr| arr.mapv(|x| F::from(x).unwrap_or(F::zero())))
                        .collect())
                        "TPU runtime not available".to_string(),
            OptimizationChoice::CPUSerial => {
                // Fallback to simple implementation
                self.execute_cpu_serial(operation_type, inputs)
            OptimizationChoice::Hybrid(strategies) => {
                // Try strategies in order until one succeeds
                let mut last_error = None;
                for strategy in strategies {
                    match self.execute_with_strategy(strategy, operation_type, inputs_context) {
                        Ok(result) => return Ok(result),
                        Err(err) => last_error = Some(err),
                Err(last_error.unwrap_or_else(|| {
                    NeuralError::NotImplemented("No hybrid strategies provided".to_string())
                }))
            OptimizationChoice::GPU => Err(NeuralError::NotImplemented(
                "GPU acceleration not yet implemented".to_string(),
            )),
    /// Select fastest available strategy
    fn select_fastest_strategy(&self, context: &OperationContext) -> Result<OptimizationChoice> {
        // Check historical performance data
        if let Ok(monitor) = self.monitor.read() {
            let fastest = monitor.get_fastest_strategy(&context.operation_type);
            if let Some(strategy) = fastest {
                return Ok(strategy);
        // Default priority order based on general performance characteristics
        if self.tpu_runtime.is_some() && self.is_suitable_for_tpu(context) {
            Ok(OptimizationChoice::TPU)
        } else if self.jit_compiler.is_some() && self.is_suitable_for_jit(context) {
            Ok(OptimizationChoice::JIT)
        } else if self.is_suitable_for_parallel(context) {
            Ok(OptimizationChoice::CPUParallel)
        } else {
            Ok(OptimizationChoice::CPUSerial)
    /// Select energy efficient strategy
    fn select_energy_efficient_strategy(
        // TPU is generally most energy efficient for large operations
        if self.tpu_runtime.is_some() && self.is_large_operation(context) {
            // For smaller operations, CPU SIMD is often more energy efficient
    /// Select balanced strategy
    fn select_balanced_strategy(&self, context: &OperationContext) -> Result<OptimizationChoice> {
        // Balance between performance and energy consumption
        let operation_size = self.estimate_operation_size(context);
        if operation_size > 1_000_000 && self.tpu_runtime.is_some() {
        } else if operation_size > 100_000 && self.jit_compiler.is_some() {
    /// Select strategy using adaptive learning
    fn select_adaptive_strategy(
            // Use recent performance history to select strategy
            let recent_performance =
                monitor.get_recent_performance(&context.operation_type, window_size);
            if let Some(best_strategy) = recent_performance {
                return Ok(best_strategy);
        // Fallback to balanced strategy
        self.select_balanced_strategy(context)
    /// Helper functions for strategy selection
    fn is_suitable_for_tpu(&self, context: &OperationContext) -> bool {
        // TPU is good for large tensor operations
        self.estimate_operation_size(context) > 1_000_000
    fn is_suitable_for_jit(&self, context: &OperationContext) -> bool {
        // JIT is good for medium-sized operations with repeated patterns
        let size = self.estimate_operation_size(context);
        size > 10_000 && size < 10_000_000
    fn is_suitable_for_parallel(&self, context: &OperationContext) -> bool {
        // Parallel processing is good for most operations
        self.estimate_operation_size(context) > 1000
    fn is_large_operation(&self, context: &OperationContext) -> bool {
    fn estimate_operation_size(&self, context: &OperationContext) -> usize {
        // Rough estimate based on input/output sizes
        let input_size: usize = context
            .inputshapes
            .iter()
            .map(|shape| shape.iter().product::<usize>())
            .sum();
        let output_size: usize = context
            .outputshapes
        input_size + output_size
    /// Convert operation to JIT format
    fn convert_to_jit_operation<F: Float + Debug>(
    ) -> Result<JITOperation> {
        match operation_type {
            "matmul" => {
                if inputs.len() >= 2 {
                    Ok(JITOperation::MatMul {
                        ashape: inputs[0].shape().to_vec(),
                        bshape: inputs[1].shape().to_vec(),
                        transpose_a: false,
                        transpose_b: false,
                    })
                    Err(NeuralError::InvalidArgument(
                        "MatMul requires 2 inputs".to_string(),
            "elementwise_add" => {
                    Ok(JITOperation::ElementwiseAdd {
                        shape: inputs[0].shape().to_vec(),
                        "ElementwiseAdd requires 2 inputs".to_string(),
            "relu" => {
                if !inputs.is_empty() {
                    Ok(JITOperation::ReLU {
                        "ReLU requires 1 input".to_string(),
            "conv2d" => {
                    Ok(JITOperation::Conv2D {
                        inputshape: inputs[0].shape().to_vec(),
                        kernelshape: inputs[1].shape().to_vec(),
                        stride: (1, 1),
                        padding: (0, 0),
                        "Conv2D requires 2 inputs".to_string(),
            "batch_norm" => {
                    Ok(JITOperation::BatchNorm {
                        eps: 1e-5,
                        "BatchNorm requires 1 input".to_string(),
            "softmax" => {
                    Ok(JITOperation::Softmax {
                        axis: -1,
                        "Softmax requires 1 input".to_string(, _ => Err(NeuralError::NotImplemented(format!(
                "JIT operation {} not supported",
                operation_type
            ))),
    /// Convert operation to TPU format
    fn convert_to_tpu_operation<F: Float + Debug>(
    ) -> Result<TPUOperation<F>> {
            "matmul" => Ok(TPUOperation::MatMul {
                transpose_a: false,
                transpose_b: false,
            }),
            "elementwise_add" => Ok(TPUOperation::ElementwiseAdd),
            "relu" => Ok(TPUOperation::ReLU),
            "conv2d" => Ok(TPUOperation::Conv2D {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
            "batch_norm" => Ok(TPUOperation::BatchNorm {
                eps: 1e-5,
                momentum: 0.1,
            "softmax" => Ok(TPUOperation::Softmax { axis: -1 }),
            "reduce_sum" => Ok(TPUOperation::ReduceSum {
                axis: None,
                keepdims: false,
            "reduce_mean" => Ok(TPUOperation::ReduceMean {
            "transpose" => Ok(TPUOperation::Transpose { axes: None }),
            "reshape" => Ok(TPUOperation::Reshape),
                "TPU operation {} not supported",
    /// Infer output shapes for JIT operations
    fn infer_jit_outputshapes<F: Float + Debug>(
        operation: &JITOperation,
    ) -> Result<Vec<Vec<usize>>> {
        match operation {
            JITOperation::MatMul {
                ashape, bshape, ..
            } => Ok(vec![vec![ashape[0], bshape[1]]], _ => {
                    Ok(vec![inputs[0].shape().to_vec()])
                        "Cannot infer output shapes without inputs".to_string(),
    /// Execute operation using CPU serial implementation
    fn execute_cpu_serial<F: Float + Debug>(
                    let result = self.serial_matmul(inputs[0], inputs[1])?;
                    Ok(vec![result])
                    let result = self.serial_elementwise_add(inputs[0], inputs[1])?;
                    let result = self.serial_relu(inputs[0]);
            "reduce_sum" => {
                    let result = self.serial_reduce_sum(inputs[0]);
                        "ReduceSum requires 1 input".to_string(),
            "reduce_mean" => {
                    let result = self.serial_reduce_mean(inputs[0]);
                        "ReduceMean requires 1 input".to_string(),
                "Serial operation {} not supported",
    /// Simple serial matrix multiplication
    fn serial_matmul<F: Float + Debug>(&self, a: &ArrayD<F>, b: &ArrayD<F>) -> Result<ArrayD<F>> {
        if a.ndim() != 2 || b.ndim() != 2 {
            return Err(NeuralError::InvalidArgument(
                "Matrix multiplication requires 2D arrays".to_string(),
            ));
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        if k != b.shape()[0] {
            return Err(NeuralError::DimensionMismatch(
                "Matrix dimensions don't match for multiplication".to_string(),
        let mut result = scirs2_core::ndarray::Array::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = F::zero();
                for ki in 0..k {
                    sum = sum + a[[i, ki]] * b[[ki, j]];
                result[[i, j]] = sum;
        Ok(result.into_dyn())
    /// Simple serial elementwise addition
    fn serial_elementwise_add<F: Float + Debug>(
        a: &ArrayD<F>,
        b: &ArrayD<F>,
    ) -> Result<ArrayD<F>> {
        if a.shape() != b.shape() {
                "Arrays must have the same shape for elementwise addition".to_string(),
        let mut result = a.clone();
        for (r, b_val) in result.iter_mut().zip(b.iter()) {
            *r = *r + *b_val;
        Ok(result)
    /// Simple serial ReLU activation
    fn serial_relu<F: Float + Debug>(&self, input: &ArrayD<F>) -> ArrayD<F> {
        input.mapv(|x| if x > F::zero() { x } else { F::zero() })
    /// Simple serial reduction sum
    fn serial_reduce_sum<F: Float + Debug + Sum>(&self, input: &ArrayD<F>) -> ArrayD<F> {
        let sum_value = input.iter().copied().sum();
        scirs2_core::ndarray::arr0(sum_value).into_dyn()
    /// Simple serial reduction mean
    fn serial_reduce_mean<F: Float + Debug + Sum + Div<Output = F>>(
        input: &ArrayD<F>,
    ) -> ArrayD<F> {
        let sum_value: F = input.iter().copied().sum();
        let count = F::from(input.len()).unwrap_or_else(|| F::one());
        let mean_value = sum_value / count;
        scirs2_core::ndarray::arr0(mean_value).into_dyn()
    /// Helper functions for caching and monitoring
    fn create_operation_key<F: Float + Debug>(
    ) -> OperationKey {
        let shape_signature = inputs
            .map(|input| format!("{:?}", input.shape()))
            .collect::<Vec<_>>()
            .join(";");
        let param_signature = format!("batch, _size:{}", context.batch_size);
        OperationKey {
            operation_type: operation_type.to_string(),
            shape_signature,
            param_signature,
    fn get_cached_optimization(&self, key: &OperationKey) -> Option<OptimizationChoice> {
        if let Ok(cache) = self.operation_cache.read() {
            cache.get(key).cloned()
            None
    fn update_operation_cache(&self, key: OperationKey, choice: OptimizationChoice) {
        if let Ok(mut cache) = self.operation_cache.write() {
            cache.insert(key, choice);
    fn record_performance_sample(&self, sample: PerformanceSample) {
        if let Ok(mut monitor) = self.monitor.write() {
            monitor.record_sample(sample);
    fn estimate_memory_usage<F: Float + Debug>(&self, inputs: &[&ArrayD<F>]) -> usize {
        inputs
            .map(|input| input.len() * std::mem::size_of::<F>())
            .sum()
    fn calculate_conv2d_outputshape(
    ) -> Vec<usize> {
        let n = input.shape()[0];
        let c_out = kernel.shape()[0];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];
        let kh = kernel.shape()[2];
        let kw = kernel.shape()[3];
        let h_out = (h_in + 2 * padding.0 - kh) / stride.0 + 1;
        let w_out = (w_in + 2 * padding.1 - kw) / stride.1 + 1;
        vec![n, c_out, h_out, w_out]
    /// Detect GPU availability (placeholder)
    fn detect_gpu_availability() -> bool {
        // Check for NVIDIA CUDA
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/usr/local/cuda").exists()
            || std::env::var("CUDA_HOME").is_ok()
        {
            return true;
        // Check for AMD ROCm
        if std::env::var("HIP_VISIBLE_DEVICES").is_ok()
            || std::path::Path::new("/opt/rocm").exists()
            || std::env::var("ROCM_PATH").is_ok()
        // Check for Intel GPU
        if std::env::var("ZE_AFFINITY_MASK").is_ok() {
        // Check for Apple Metal
        #[cfg(target_os = "macos")]
            // On macOS, Metal is available on most systems
        // Check for OpenCL devices
        Self::detect_opencl_devices()
    /// Detect OpenCL devices
    fn detect_opencl_devices() -> bool {
        // Check for OpenCL installation
        if std::path::Path::new("/usr/lib/libOpenCL.so").exists()
            || std::path::Path::new("/usr/lib64/libOpenCL.so").exists()
            || std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
        // Check environment variables
        std::env::var("OPENCL_VENDOR_PATH").is_ok()
    /// Get comprehensive performance statistics
    pub fn get_unified_stats(&self) -> UnifiedPerformanceStats {
        let cpu_stats = self.cpu_optimizer.get_performance_stats();
        let jit_stats = if let Some(jit_compiler) = &self.jit_compiler {
            let jit_raw_stats = jit_compiler.get_statistics();
            Some(JITStats {
                kernels_compiled: jit_raw_stats.kernels_compiled,
                cache_hit_rate: jit_raw_stats.cache_hit_rate,
                avg_compile_time_ms: jit_raw_stats.avg_compile_time_ms,
                total_execution_time_ms: jit_raw_stats.total_execution_time_ms,
            })
        let tpu_stats = if let Some(tpu_runtime) = &self.tpu_runtime {
            let tpu_raw_stats = tpu_runtime.get_statistics();
            Some(TPUStats {
                operations_executed: tpu_raw_stats.total_operations,
                avg_execution_time_ms: tpu_raw_stats.avg_ops_per_second.recip() * 1000.0,
                avg_utilization: 0.8, // Placeholder
                peak_memory_usage: tpu_raw_stats.total_data_processed as usize,
        let unified_stats = if let Ok(monitor) = self.monitor.read() {
            monitor.get_unified_stats()
            UnifiedStats::default()
        UnifiedPerformanceStats {
            cpu_stats,
            jit_stats,
            tpu_stats,
            gpu_stats: None,
            unified_stats,
    /// Reset all performance tracking
    pub fn reset_performance_tracking(&mut self) {
        self.cpu_optimizer.reset_stats();
            monitor.reset();
            cache.clear();
    /// Set auto-optimization strategy
    pub fn set_auto_optimization_strategy(&mut self, strategy: AutoOptimizationStrategy) {
        self.auto_optimization = strategy;
    /// Create a hybrid strategy that tries multiple optimization approaches
    pub fn create_hybrid_strategy(&self, primary: OptimizationChoice) -> OptimizationChoice {
        let fallback_strategies = match primary {
            OptimizationChoice::TPU => vec![
                OptimizationChoice::TPU,
                OptimizationChoice::JIT,
                OptimizationChoice::CPUSimd,
                OptimizationChoice::CPUParallel,
                OptimizationChoice::CPUSerial,
            ],
            OptimizationChoice::JIT => vec![
            OptimizationChoice::CPUSimd => vec![
            OptimizationChoice::CPUParallel => vec![
            _ => vec![primary.clone(), OptimizationChoice::CPUSerial],
        OptimizationChoice::Hybrid(fallback_strategies)
    /// Execute with automatic fallback on failure
    pub fn execute_with_fallback<F: Float + Debug>(
        preferred_strategy: OptimizationChoice,
        let hybrid_strategy = self.create_hybrid_strategy(preferred_strategy);
        self.execute_optimized(operation_type, inputs, context.clone())
            .or_else(|_| {
                // If auto optimization fails, try the hybrid approach
                self.execute_with_strategy(&hybrid_strategy, operation_type, inputs, &context)
impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
            memory_usage: HashMap::new(),
            success_rates: HashMap::new(),
            device_utilization: HashMap::new(),
            performance_trends: Vec::new(),
            total_operations: 0,
    /// Record a performance sample
    pub fn record_sample(&mut self, sample: PerformanceSample) {
        self.total_operations += 1;
        // Update execution times
        self.execution_times
            .entry(sample.strategy.clone())
            .or_insert_with(Vec::new)
            .push(sample.execution_time);
        // Update memory usage
        self.memory_usage
            .push(sample.memory_usage);
        // Update success rates
        let (successes, total) = self
            .success_rates
            .or_insert((0, 0));
        if sample.success {
            *successes += 1;
        *total += 1;
        // Store trend data (keep last 1000 samples)
        self.performance_trends.push(sample);
        if self.performance_trends.len() > 1000 {
            self.performance_trends.drain(0..100); // Remove oldest 100
    /// Get fastest strategy for an operation type
    pub fn get_fastest_strategy(&self, operationtype: &str) -> Option<OptimizationChoice> {
        let relevant_samples: Vec<_> = self
            .performance_trends
            .filter(|sample| sample.operation == operation_type && sample.success)
            .collect();
        if relevant_samples.is_empty() {
            return None;
        // Group by strategy and find average execution time
        let mut strategy_times: HashMap<OptimizationChoice, Vec<Duration>> = HashMap::new();
        for sample in relevant_samples {
            strategy_times
                .entry(sample.strategy.clone())
                .or_insert_with(Vec::new)
                .push(sample.execution_time);
        // Find strategy with lowest average time
        strategy_times
            .into_iter()
            .min_by_key(|(_, times)| {
                let avg = times.iter().sum::<Duration>() / times.len() as u32;
                avg
            .map(|(strategy_)| strategy)
    /// Get recent performance for adaptive strategy
    pub fn get_recent_performance(
    ) -> Option<OptimizationChoice> {
        let recent_samples: Vec<_> = self
            .rev()
            .take(window_size)
        if recent_samples.is_empty() {
        // Return the strategy with best recent performance
        let mut strategy_performance: HashMap<OptimizationChoice, f64> = HashMap::new();
        for sample in recent_samples {
            let score = 1.0 / sample.execution_time.as_secs_f64(); // Higher is better
            *strategy_performance
                .or_insert(0.0) += score;
        strategy_performance
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    /// Get unified statistics
    pub fn get_unified_stats(&self) -> UnifiedStats {
        let total_operations = self.total_operations;
        // Calculate average speedup (simplified)
        let avg_speedup = 1.5; // Placeholder calculation
        // Calculate energy efficiency (simplified)
        let energy_efficiency = 1.3; // Placeholder calculation
        // Find preferred strategy
        let preferred_strategy = self
            .execution_times
            .max_by_key(|(_, times)| times.len())
            .map(|(strategy_)| strategy.clone())
            .unwrap_or(OptimizationChoice::CPUParallel);
        // Calculate strategy distribution
        let mut strategy_distribution = HashMap::new();
        let total_samples = self.performance_trends.len() as f64;
        if total_samples > 0.0 {
            for sample in &self.performance_trends {
                *strategy_distribution
                    .entry(sample.strategy.clone())
                    .or_insert(0.0) += 1.0 / total_samples;
        UnifiedStats {
            total_operations,
            avg_speedup,
            energy_efficiency,
            preferred_strategy,
            strategy_distribution,
    /// Reset monitoring data
    pub fn reset(&mut self) {
        self.execution_times.clear();
        self.memory_usage.clear();
        self.success_rates.clear();
        self.device_utilization.clear();
        self.performance_trends.clear();
        self.total_operations = 0;
impl Default for UnifiedStats {
    fn default() -> Self {
            avg_speedup: 1.0,
            energy_efficiency: 1.0,
            preferred_strategy: OptimizationChoice::CPUParallel,
            strategy_distribution: HashMap::new(),
impl Default for UnifiedPerformanceManager {
        Self::new().unwrap_or_else(|_| {
            // Fallback to minimal manager with basic CPU configuration
            use std::collections::HashMap;
            UnifiedPerformanceManager {
                cpu_optimizer: None, // Will use basic CPU operations
                jit_compiler: None,
                tpu_runtime: None,
                gpu_accelerator: None,
                metrics: UnifiedMetrics::default(),
                preferred_strategy: OptimizationChoice::CPUSerial,
                strategy_distribution: HashMap::new(),
impl std::fmt::Display for OptimizationChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationChoice::CPUSimd => write!(f, "CPU SIMD"),
            OptimizationChoice::CPUParallel => write!(f, "CPU Parallel"),
            OptimizationChoice::CPUSerial => write!(f, "CPU Serial"),
            OptimizationChoice::JIT => write!(f, "JIT Compiled"),
            OptimizationChoice::TPU => write!(f, "TPU"),
            OptimizationChoice::GPU => write!(f, "GPU"),
                write!(
                    f,
                    "Hybrid({})",
                    strategies
                        .iter()
                        .map(|s| format!("{}", s))
                        .collect::<Vec<_>>()
                        .join("+")
                )
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;
    #[test]
    fn test_unified_performance_manager_creation() {
        let manager = UnifiedPerformanceManager::new();
        assert!(manager.is_ok());
    fn test_operation_context_creation() {
            inputshapes: vec![vec![100, 200], vec![200, 150]],
            outputshapes: vec![vec![100, 150]],
            memory_limit: Some(1024 * 1024),
            time_limit: Some(1000),
            energy_limit: Some(0.8),
            batch_size: 100,
        assert_eq!(context.operation_type, "matmul");
        assert_eq!(context.batch_size, 100);
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        let sample = PerformanceSample {
            timestamp: Instant::now(),
            operation: "test_op".to_string(),
            strategy: OptimizationChoice::CPUParallel,
            execution_time: Duration::from_millis(100),
            memory_usage: 1024,
            success: true,
        monitor.record_sample(sample);
        assert_eq!(monitor.total_operations, 1);
    fn test_optimization_choice_display() {
        assert_eq!(
            format!("{}", OptimizationChoice::CPUParallel),
            "CPU Parallel"
        );
        assert_eq!(format!("{}", OptimizationChoice::JIT), "JIT Compiled");
        assert_eq!(format!("{}", OptimizationChoice::TPU), "TPU");
    fn test_operation_size_estimation() {
        let manager = UnifiedPerformanceManager::new().unwrap();
        let size = manager.estimate_operation_size(&context);
        assert_eq!(size, 100 * 200 + 200 * 150 + 100 * 150); // inputs + outputs
    fn test_serial_matmul() {
        let a = Array::ones((3, 4)).into_dyn();
        let b = Array::ones((4, 5)).into_dyn();
        let result = manager.serial_matmul(&a, &b);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.shape(), &[3, 5]);
        assert_eq!(result[[0, 0]], 4.0); // sum of 1*1 for 4 elements
