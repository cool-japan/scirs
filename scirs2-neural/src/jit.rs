//! Just-In-Time (JIT) compilation system for neural networks
//!
//! This module provides JIT compilation capabilities for optimizing neural network
//! operations at runtime. It includes:
//! - Dynamic code generation for optimized kernels
//! - Operation fusion for reducing memory overhead
//! - Platform-specific optimizations
//! - SIMD and vectorization hints
//! - Cache-friendly memory access patterns

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{par_azip, s, Array, ArrayD, ArrayView, ArrayViewMut, Dimension};
use scirs2_core::numeric::Float;
use scirs2_core::parallel_ops::*;
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
/// JIT compilation target architecture
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum TargetArchitecture {
    /// x86-64 with specific feature sets
    X86_64 {
        /// AVX support level (0, 1, 2, 512)
        avx_level: u8,
        /// FMA support
        fma: bool,
        /// BMI support
        bmi: bool,
    },
    /// ARM64 with NEON support
    ARM64 {
        /// NEON support
        neon: bool,
        /// SVE support
        sve: bool,
    /// RISC-V with vector extensions
    RISCV {
        /// Vector extension support
        vector: bool,
    /// GPU targets
    GPU {
        /// GPU architecture type
        arch: GPUArchitecture,
    /// WebAssembly target
    WASM {
        /// SIMD support
        simd: bool,
        /// Threading support
        threads: bool,
    /// Generic fallback
    Generic,
}
/// GPU architecture types
pub enum GPUArchitecture {
    /// NVIDIA CUDA
    CUDA {
        /// Compute capability
        compute_capability: (u8, u8),
    /// AMD ROCm
    ROCm {
        /// GFX version
        gfx_version: String,
    /// Intel GPU
    Intel {
        /// Generation
        generation: u8,
    /// Apple Metal
    Metal {
        /// Family
        family: u8,
    /// Vulkan compute
    Vulkan,
    /// OpenCL
    OpenCL {
        /// Version
        version: String,
/// JIT operation types that can be compiled
pub enum JITOperation {
    /// Matrix multiplication with specific dimensions
    MatMul {
        /// Input A dimensions
        ashape: Vec<usize>,
        /// Input B dimensions
        bshape: Vec<usize>,
        /// Transpose flags
        transpose_a: bool,
        transpose_b: bool,
    /// Element-wise operations
    ElementWise {
        /// Operation type
        op: ElementWiseOp,
        /// Input shapes
        shapes: Vec<Vec<usize>>,
    /// Convolution operations
    Convolution {
        /// Input shape [N, C, H, W]
        inputshape: Vec<usize>,
        /// Weight shape [out_channels, in_channels, kH, kW]
        weightshape: Vec<usize>,
        /// Stride
        stride: Vec<usize>,
        /// Padding
        padding: Vec<usize>,
        /// Dilation
        dilation: Vec<usize>,
    /// Pooling operations
    Pooling {
        /// Pooling type
        pool_type: PoolingType,
        /// Input shape
        /// Kernel size
        kernel_size: Vec<usize>,
    /// Normalization operations
    Normalization {
        /// Normalization type
        norm_type: NormalizationType,
        /// Axes to normalize over
        axes: Vec<usize>,
    /// Activation functions
    Activation {
        /// Activation type
        activation: ActivationType,
    /// Reduction operations
    Reduction {
        /// Reduction type
        reduction: ReductionType,
        /// Reduction axes
        /// Keep dimensions
        keep_dims: bool,
    /// Custom fused operations
    FusedOp {
        /// Sequence of operations to fuse
        operations: Vec<Box<JITOperation>>,
        /// Fusion strategy
        fusion_strategy: FusionStrategy,
/// Element-wise operation types
pub enum ElementWiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
    Equal,
    Greater,
    Less,
    Abs,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    /// Custom operation with code
    Custom(String),
/// Pooling operation types
pub enum PoolingType {
    Average,
    Global,
    Adaptive,
/// Normalization types
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    InstanceNorm,
    GroupNorm,
/// Activation function types
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Softmax,
    GELU,
    Swish,
    Mish,
    LeakyReLU(f64),
    ELU(f64),
    SELU,
/// Reduction operation types
pub enum ReductionType {
    Sum,
    Mean,
    Prod,
    Std,
    Var,
/// Fusion strategies for combining operations
pub enum FusionStrategy {
    /// Vertical fusion (producer-consumer)
    Vertical,
    /// Horizontal fusion (independent operations)
    Horizontal,
    /// Loop fusion
    Loop,
    /// Memory fusion (share memory buffers)
    Memory,
/// JIT compilation context and cache
pub struct JITCompiler {
    /// Target architecture for compilation
    target_arch: TargetArchitecture,
    /// Compiled kernel cache
    kernel_cache: Arc<RwLock<HashMap<JITKernelKey, CompiledKernel>>>,
    /// Optimization settings
    optimization_level: OptimizationLevel,
    /// Code generation settings
    codegen_settings: CodeGenSettings,
    /// Runtime statistics
    stats: Arc<RwLock<JITStatistics>>,
    /// Operation fusion optimizer
    fusion_optimizer: FusionOptimizer,
/// Key for identifying compiled kernels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JITKernelKey {
    /// Operation description
    operation: JITOperation,
    /// Target architecture
    target: TargetArchitecture,
    /// Optimization level
    opt_level: OptimizationLevel,
/// Compiled kernel with metadata
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Generated code (platform-specific)
    code: String,
    /// Entry point function name
    entry_point: String,
    /// Compiled function pointer (for native execution)
    compiled_fn: Option<CompiledFunction>,
    /// Memory requirements
    memory_requirements: MemoryRequirements,
    /// Performance characteristics
    performance_hints: PerformanceHints,
    /// Compilation timestamp
    timestamp: std::time::Instant,
    /// Usage count
    usage_count: Arc<AtomicU64>,
    /// Kernel optimization metadata
    optimization_metadata: OptimizationMetadata,
/// Compiled function types for different operations
pub enum CompiledFunction {
    /// Element-wise operation function
    ElementWise(ElementWiseKernel),
    /// Matrix multiplication function
    MatMul(MatMulKernel),
    /// Convolution function
    Convolution(ConvolutionKernel),
    /// Activation function
    Activation(ActivationKernel),
    /// Fused operation
    Fused(FusedKernel),
/// Element-wise operation kernel
pub struct ElementWiseKernel {
    pub operation: ElementWiseOp,
    pub input_count: usize,
    pub use_simd: bool,
/// Matrix multiplication kernel
pub struct MatMulKernel {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub block_size: usize,
/// Convolution kernel
pub struct ConvolutionKernel {
    pub inputshape: Vec<usize>,
    pub weightshape: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
    pub use_im2col: bool,
/// Activation function kernel
pub struct ActivationKernel {
    pub activation: ActivationType,
    pub use_lookup_table: bool,
/// Fused operation kernel
pub struct FusedKernel {
    pub operations: Vec<CompiledFunction>,
    pub fusion_strategy: FusionStrategy,
    pub buffer_reuse: bool,
/// Optimization metadata for kernels
pub struct OptimizationMetadata {
    /// SIMD instruction set used
    pub simd_level: SimdLevel,
    /// Memory access pattern optimization
    pub memory_optimization: MemoryOptimization,
    /// Loop unrolling factor
    pub unroll_factor: u8,
    /// Parallelization strategy
    pub parallel_strategy: ParallelStrategy,
    /// Cache blocking parameters
    pub cache_blocks: Vec<usize>,
/// SIMD instruction levels
#[derive(Debug, Clone, PartialEq)]
pub enum SimdLevel {
    None,
    SSE,
    AVX,
    AVX2,
    AVX512,
    NEON,
    SVE,
/// Memory optimization strategies
pub enum MemoryOptimization {
    Prefetch,
    Streaming,
    Blocking,
    TileOptimized,
/// Parallelization strategies
pub enum ParallelStrategy {
    OpenMP,
    Rayon,
    SIMD,
    Hybrid,
/// Memory requirements for a kernel
pub struct MemoryRequirements {
    /// Minimum required memory (bytes)
    min_memory: usize,
    /// Optimal memory for performance (bytes)
    optimal_memory: usize,
    /// Memory access pattern
    access_pattern: MemoryAccessPattern,
    /// Alignment requirements
    alignment: usize,
/// Memory access patterns
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
    Blocked,
/// Performance hints for kernel execution
pub struct PerformanceHints {
    /// Estimated FLOPS
    estimated_flops: u64,
    /// Memory bandwidth utilization (0-1)
    memory_bandwidth_util: f64,
    /// Compute intensity (FLOPS per byte)
    compute_intensity: f64,
    /// Vectorization factor
    vectorization_factor: u8,
    /// Parallelization level
    parallelization_level: u8,
/// JIT compilation optimization levels
pub enum OptimizationLevel {
    /// No optimization, fast compilation
    O0,
    /// Basic optimizations
    O1,
    /// Standard optimizations
    O2,
    /// Aggressive optimizations
    O3,
    /// Size optimization
    Os,
    /// Custom optimization with specific flags
    Custom(Vec<String>),
/// Code generation settings
pub struct CodeGenSettings {
    /// Enable vectorization
    pub vectorize: bool,
    /// Unroll loops
    pub unroll_loops: bool,
    /// Use lookup tables for functions
    pub use_lookup_tables: bool,
    /// Inline functions aggressively
    pub aggressive_inlining: bool,
    /// Use platform-specific intrinsics
    pub use_intrinsics: bool,
    /// Generate debug information
    pub debug_info: bool,
    /// Target specific features
    pub target_features: HashSet<String>,
/// JIT compilation and execution statistics
#[derive(Debug, Clone, Default)]
pub struct JITStatistics {
    /// Number of kernels compiled
    pub kernels_compiled: u64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total compilation time (ms)
    pub total_compile_time_ms: f64,
    /// Total execution time (ms)
    pub total_execution_time_ms: f64,
    /// Average compilation time per kernel (ms)
    pub avg_compile_time_ms: f64,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Most frequently used operations
    pub popular_operations: HashMap<String, u64>,
/// Operation fusion optimizer
pub struct FusionOptimizer {
    /// Fusion rules database
    fusion_rules: HashMap<(String, String), FusionRule>,
    /// Maximum fusion depth
    max_fusion_depth: usize,
    /// Memory threshold for fusion
    memory_threshold: usize,
/// Fusion rule for combining operations
pub struct FusionRule {
    /// Can these operations be fused?
    pub can_fuse: bool,
    /// Expected performance improvement (ratio)
    pub performance_gain: f64,
    /// Memory savings (bytes)
    pub memory_savings: usize,
    /// Fusion strategy to use
    pub strategy: FusionStrategy,
impl JITCompiler {
    /// Create a new JIT compiler for the target architecture
    pub fn new(_targetarch: TargetArchitecture) -> Self {
        let codegen_settings = CodeGenSettings {
            vectorize: true,
            unroll_loops: true,
            use_lookup_tables: false,
            aggressive_inlining: true,
            use_intrinsics: true,
            debug_info: false,
            target_features: HashSet::new(),
        };
        let fusion_optimizer = FusionOptimizer::new();
        Self {
            target_arch,
            kernel_cache: Arc::new(RwLock::new(HashMap::new())),
            optimization_level: OptimizationLevel::O2,
            codegen_settings,
            stats: Arc::new(RwLock::new(JITStatistics::default())),
            fusion_optimizer,
        }
    }
    /// Detect the best target architecture for the current system
    pub fn detect_target_architecture() -> TargetArchitecture {
        #[cfg(target_arch = "x86_64")]
        {
            TargetArchitecture::X86_64 {
                avx_level: detect_avx_level(),
                fma: is_x86_feature_detected!("fma"),
                bmi: is_x86_feature_detected!("bmi1"),
            }
        #[cfg(target_arch = "aarch64")]
            TargetArchitecture::ARM64 {
                neon: true, // NEON is standard on ARM64
                sve: false, // SVE detection would require runtime checks
        #[cfg(target_arch = "wasm32")]
            TargetArchitecture::WASM {
                simd: true,     // Assume SIMD support in modern browsers
                threads: false, // Threading support varies
        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "wasm32"
        )))]
            TargetArchitecture::Generic
    /// Compile a JIT operation to optimized code
    pub fn compile_operation(&self, operation: &JITOperation) -> Result<CompiledKernel> {
        let start_time = std::time::Instant::now();
        // Create kernel key
        let key = JITKernelKey {
            operation: operation.clone(),
            target: self.target_arch.clone(),
            opt_level: self.optimization_level.clone(),
        // Check cache first
        if let Some(cached_kernel) = self.get_cached_kernel(&key) {
            self.update_cache_stats(true);
            return Ok(cached_kernel);
        // Apply fusion optimization if applicable
        let optimized_operation = self.fusion_optimizer.optimize_operation(operation)?;
        // Generate code for the operation
        let code = self.generate_code(&optimized_operation)?;
        let kernel_id = self.generate_kernel_id(&key);
        let entry_point = format!("kernel_{kernel_id}");
        // Analyze memory requirements
        let memory_requirements = self.analyze_memory_requirements(&optimized_operation)?;
        // Estimate performance characteristics
        let performance_hints = self.estimate_performance(&optimized_operation)?;
        // Compile the operation to actual executable code
        let compiled_fn = self.compile_to_native(&optimized_operation)?;
        // Generate optimization metadata
        let optimization_metadata = self.generate_optimization_metadata(&optimized_operation)?;
        let kernel = CompiledKernel {
            code,
            entry_point,
            compiled_fn: Some(compiled_fn),
            memory_requirements,
            performance_hints,
            timestamp: std::time::Instant::now(),
            usage_count: Arc::new(AtomicU64::new(0)),
            optimization_metadata,
        // Cache the compiled kernel
        self.cache_kernel(key, kernel.clone());
        // Update statistics
        let compile_time = start_time.elapsed().as_millis() as f64;
        self.update_compile_stats(compile_time);
        self.update_cache_stats(false);
        Ok(kernel)
    /// Execute a compiled kernel with given inputs
    pub fn execute_kernel<F: Float + Debug + 'static>(
        &self,
        kernel: &CompiledKernel,
        inputs: &[&ArrayD<F>],
        outputshape: &[usize],
    ) -> Result<ArrayD<F>> {
        // Validate inputs
        self.validate_kernel_inputs(kernel, inputs)?;
        // Execute using the enhanced native implementation
        let output = self.execute_kernel_native(kernel, inputs, outputshape)?;
        // Update usage statistics
        kernel.usage_count.fetch_add(1, Ordering::Relaxed);
        let execution_time = start_time.elapsed().as_millis() as f64;
        self.update_execution_stats(execution_time);
        Ok(output)
    /// Compile and execute an operation in one step
    pub fn compile_and_execute<F: Float + Debug + 'static>(
        operation: &JITOperation,
        let kernel = self.compile_operation(operation)?;
        self.execute_kernel(&kernel, inputs, outputshape)
    /// Get cached kernel if available
    fn get_cached_kernel(&self, key: &JITKernelKey) -> Option<CompiledKernel> {
        if let Ok(cache) = self.kernel_cache.read() {
            cache.get(key).cloned()
        } else {
            None
    /// Cache a compiled kernel
    fn cache_kernel(&self, key: JITKernelKey, kernel: CompiledKernel) {
        if let Ok(mut cache) = self.kernel_cache.write() {
            cache.insert(key, kernel);
    /// Generate unique kernel ID
    fn generate_kernel_id(&self, key: &JITKernelKey) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{hash:x}")
    /// Generate optimized code for an operation
    fn generate_code(&self, operation: &JITOperation) -> Result<String> {
        match operation {
            JITOperation::MatMul {
                ashape,
                bshape,
                transpose_a,
                transpose_b,
            } => self.generate_matmul_code(ashape, bshape, *transpose_a, *transpose_b),
            JITOperation::ElementWise { op, shapes } => self.generate_elementwise_code(op, shapes),
            JITOperation::Convolution {
                inputshape,
                weightshape,
                stride,
                padding,
                dilation,
            } => {
                self.generate_convolution_code(inputshape, weightshape, stride, padding, dilation)
            JITOperation::Activation {
                activation,
            } => self.generate_activation_code(activation, inputshape),
            JITOperation::FusedOp {
                operations,
                fusion_strategy,
            } => self.generate_fused_code(operations, fusion_strategy, _ => {
                // For other operations, generate generic code
                Ok(self.generate_generic_code(operation))
    /// Generate matrix multiplication code
    fn generate_matmul_code(
        ashape: &[usize],
        bshape: &[usize],
    ) -> Result<String> {
        let m = if transpose_a { ashape[1] } else { ashape[0] };
        let k = if transpose_a { ashape[0] } else { ashape[1] };
        let n = if transpose_b { bshape[0] } else { bshape[1] };
        let mut code = String::new();
        code.push_str(&format!("// Optimized MatMul: {m}x{k} * {k}x{n}\n"));
        code.push_str("void kernel_matmul(const float* A, const float* B, float* C) {\n");
        if self.codegen_settings.vectorize && self.target_arch_supports_simd() {
            // Generate vectorized code
            code.push_str(&self.generate_vectorized_matmul(m, k, n)?);
            // Generate scalar code
            code.push_str(&self.generate_scalar_matmul(m, k, n, transpose_a, transpose_b));
        code.push_str("}\n");
        Ok(code)
    /// Generate element-wise operation code
    fn generate_elementwise_code(
        op: &ElementWiseOp,
        shapes: &[Vec<usize>],
        let total_elements = shapes[0].iter().product::<usize>();
        code.push_str(&format!("// Element-wise operation: {op:?}\n"));
        code.push_str("void kernel_elementwise(");
        // Generate input parameters
        for i in 0..shapes.len() {
            code.push_str(&format!("const float* input{i}, "));
        code.push_str("float* output) {\n");
            code.push_str(&self.generate_vectorized_elementwise(op, total_elements)?);
            code.push_str(&self.generate_scalar_elementwise(op, total_elements));
    /// Generate convolution code
    fn generate_convolution_code(
        inputshape: &[usize],
        weightshape: &[usize],
        stride: &[usize],
        padding: &[usize], _dilation: &[usize],
        code.push_str("// Optimized Convolution\n");
        code.push_str(
            "void kernel_conv2d(const float* input, const float* weight, float* output) {\n",
        );
        // Generate convolution loops with optimizations
        let n = inputshape[0]; // batch size
        let c_in = inputshape[1]; // input channels
        let h_in = inputshape[2]; // input height
        let w_in = inputshape[3]; // input width
        let c_out = weightshape[0]; // output channels
        let kh = weightshape[2]; // kernel height
        let kw = weightshape[3]; // kernel width
        let h_out = (h_in + 2 * padding[0] - kh) / stride[0] + 1;
        let w_out = (w_in + 2 * padding[1] - kw) / stride[1] + 1;
        if self.codegen_settings.unroll_loops && kh <= 3 && kw <= 3 {
            code.push_str(&self.generate_unrolled_conv(n, c_in, c_out, h_out, w_out, kh, kw));
            code.push_str(&self.generate_standard_conv(
                n, c_in, c_out, h_in, w_in, h_out, w_out, kh, kw, stride, padding,
            ));
    /// Generate activation function code
    fn generate_activation_code(
        activation: &ActivationType,
        let total_elements = inputshape.iter().product::<usize>();
        code.push_str(&format!("// Activation: {activation:?}\n"));
        code.push_str("void kernel_activation(const float* input, float* output) {\n");
            code.push_str(&self.generate_vectorized_activation(activation, total_elements)?);
            code.push_str(&self.generate_scalar_activation(activation, total_elements));
    /// Generate fused operation code
    fn generate_fused_code(
        operations: &[Box<JITOperation>],
        strategy: &FusionStrategy,
        code.push_str(&format!(
            "// Fused operations with strategy: {:?}\n",
            strategy
        ));
        code.push_str("void kernel_fused(");
        match strategy {
            FusionStrategy::Vertical => {
                // Generate code that combines operations in sequence
                code.push_str("const float* input, float* output) {\n");
                code.push_str("  // Vertical fusion - pipeline operations\n");
                for (i, op) in operations.iter().enumerate() {
                    code.push_str(&format!("  // Operation {i}: {op:?}\n"));
                }
            FusionStrategy::Horizontal => {
                // Generate code that combines independent operations
                code.push_str("  // Horizontal fusion - parallel operations\n");
                    code.push_str(&format!("  // Parallel operation {i}: {op:?}\n"));
                code.push_str("  // Generic fusion\n");
    /// Generate generic fallback code
    fn generate_generic_code(&self, operation: &JITOperation) -> String {
        format!("// Generic implementation for: {operation:?}\nvoid kernel_generic() {{\n  // Fallback implementation\n}}\n")
    /// Check if target architecture supports SIMD
    fn target_arch_supports_simd(&self) -> bool {
        match &self.target_arch {
            TargetArchitecture::X86_64 { avx_level, .. } => *avx_level > 0,
            TargetArchitecture::ARM64 { neon, .. } => *neon,
            TargetArchitecture::WASM { simd, .. } => *simd_ => false,
    /// Generate vectorized matrix multiplication code
    fn generate_vectorized_matmul(&self, m: usize, k: usize, n: usize) -> Result<String> {
            TargetArchitecture::X86_64 { avx_level, .. } => {
                if *avx_level >= 2 {
                    code.push_str(&format!("  // AVX2 vectorized matmul {}x{}x{}\n", m, k, n));
                    code.push_str("  #pragma omp parallel for\n");
                    code.push_str("  for (int i = 0; i < m; i += 8) {\n");
                    code.push_str("    for (int j = 0; j < n; j += 8) {\n");
                    code.push_str("      __m256 sum = _mm256_setzero_ps();\n");
                    code.push_str("      for (int l = 0; l < k; l++) {\n");
                    code.push_str("        __m256 a_vec = _mm256_broadcast_ss(&A[i*k + l]);\n");
                    code.push_str("        __m256 b_vec = _mm256_loadu_ps(&B[l*n + j]);\n");
                    code.push_str("        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);\n");
                    code.push_str("      }\n");
                    code.push_str("      _mm256_storeu_ps(&C[i*n + j], sum);\n");
                    code.push_str("    }\n");
                    code.push_str("  }\n");
                } else {
                    code.push_str("  // SSE vectorized matmul\n");
                    code.push_str(&self.generate_sse_matmul(m, k, n));
            TargetArchitecture::ARM64 { .. } => {
                code.push_str("  // NEON vectorized matmul\n");
                code.push_str(&self.generate_neon_matmul(m, k, n));
                return Err(NeuralError::ComputationError(
                    "Vectorization not supported for this architecture".to_string(),
                ));
    /// Generate scalar matrix multiplication code
    fn generate_scalar_matmul(
        m: usize,
        k: usize,
        n: usize,
    ) -> String {
        code.push_str(&format!("  // Scalar matmul {}x{}x{}\n", m, k, n));
        code.push_str("  #pragma omp parallel for\n");
        code.push_str("  for (int i = 0; i < m; i++) {\n");
        code.push_str("    for (int j = 0; j < n; j++) {\n");
        code.push_str("      float sum = 0.0f;\n");
        code.push_str("      for (int l = 0; l < k; l++) {\n");
        if transpose_a && transpose_b {
            code.push_str("        sum += A[l*m + i] * B[j*k + l];\n");
        } else if transpose_a {
            code.push_str("        sum += A[l*m + i] * B[l*n + j];\n");
        } else if transpose_b {
            code.push_str("        sum += A[i*k + l] * B[j*k + l];\n");
            code.push_str("        sum += A[i*k + l] * B[l*n + j];\n");
        code.push_str("      }\n");
        code.push_str("      C[i*n + j] = sum;\n");
        code.push_str("    }\n");
        code.push_str("  }\n");
        code
    /// Generate SSE matrix multiplication code
    fn generate_sse_matmul(&self, m: usize, k: usize, n: usize) -> String {
        String::from("  // SSE implementation placeholder\n")
    /// Generate NEON matrix multiplication code
    fn generate_neon_matmul(&self, m: usize, k: usize, n: usize) -> String {
        String::from("  // NEON implementation placeholder\n")
    /// Generate vectorized element-wise code
    fn generate_vectorized_elementwise(
        total_elements: usize,
            "  // Vectorized element-wise operation, {} elements\n",
            total_elements
        code.push_str("  for (int i = 0; i < total_elements; i += 8) {\n");
        match op {
            ElementWiseOp::Add => {
                code.push_str("    __m256 a = _mm256_loadu_ps(&input0[i]);\n");
                code.push_str("    __m256 b = _mm256_loadu_ps(&input1[i]);\n");
                code.push_str("    __m256 result = _mm256_add_ps(a, b);\n");
                code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
            ElementWiseOp::Mul => {
                code.push_str("    __m256 result = _mm256_mul_ps(a, b);\n");
            ElementWiseOp::Abs => {
                code.push_str("    __m256 input_vec = _mm256_loadu_ps(&input0[i]);\n");
                code.push_str(
                    "    __m256 result = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), input_vec);\n",
                );
            ElementWiseOp::ReLU => {
                code.push_str("    __m256 zero = _mm256_setzero_ps();\n");
                code.push_str("    __m256 result = _mm256_max_ps(input_vec, zero);\n");
                code.push_str("    // Generic vectorized operation\n");
    /// Generate scalar element-wise code
    fn generate_scalar_elementwise(&self, op: &ElementWiseOp, totalelements: usize) -> String {
            "  // Scalar element-wise operation, {} elements\n",
        code.push_str("  for (int i = 0; i < total_elements; i++) {\n");
            ElementWiseOp::Add => code.push_str("    output[i] = input0[i] + input1[i];\n"),
            ElementWiseOp::Sub => code.push_str("    output[i] = input0[i] - input1[i];\n"),
            ElementWiseOp::Mul => code.push_str("    output[i] = input0[i] * input1[i];\n"),
            ElementWiseOp::Div => code.push_str("    output[i] = input0[i] / input1[i];\n"),
            ElementWiseOp::Max => code.push_str("    output[i] = fmaxf(input0[i], input1[i]);\n"),
            ElementWiseOp::Min => code.push_str("    output[i] = fminf(input0[i], input1[i]);\n"),
            ElementWiseOp::Abs => code.push_str("    output[i] = fabsf(input0[i]);\n"),
            ElementWiseOp::Sqrt => code.push_str("    output[i] = sqrtf(input0[i]);\n"),
            ElementWiseOp::Exp => code.push_str("    output[i] = expf(input0[i]);\n"),
            ElementWiseOp::Log => code.push_str("    output[i] = logf(input0[i]);\n"),
            ElementWiseOp::Sin => code.push_str("    output[i] = sinf(input0[i]);\n"),
            ElementWiseOp::Cos => code.push_str("    output[i] = cosf(input0[i]);\n"),
            ElementWiseOp::Tanh => code.push_str("    output[i] = tanhf(input0[i]);\n"),
            ElementWiseOp::Custom(expr) => {
                code.push_str(&format!(
                    "    output[i] = {};\n",
                    expr.replace("x", "input0[i]")
            _ => code.push_str("    output[i] = input0[i]; // fallback\n"),
    /// Generate unrolled convolution code
    fn generate_unrolled_conv(
        c_in: usize,
        c_out: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
        code.push_str(&format!("  // Unrolled {}x{} convolution\n", kh, kw));
        code.push_str(&format!("  for (int n = 0; n < {}; n++) {{\n", n));
            "    for (int c_out = 0; c_out < {}; c_out++) {{\n",
            c_out
        code.push_str(&format!("      for (int h = 0; h < {}; h++) {{\n", h_out));
        code.push_str(&format!("        for (int w = 0; w < {}; w++) {{\n", w_out));
        code.push_str("          float sum = 0.0f;\n");
        // Unroll the kernel loops
        for kh_i in 0..kh {
            for kw_i in 0..kw {
                    "          for (int c_in = 0; c_in < {}; c_in++) {{\n",
                    c_in
                    "            sum += input[((n*c_in + c_in)*h_in + h + {})*w_in + w + {}] * weight[((c_out*c_in + c_in)*{} + {})*{} + {}];\n",
                    kh_i, kw_i, kh, kh_i, kw, kw_i
                code.push_str("          }\n");
        code.push_str("          output[((n*c_out + c_out)*h_out + h)*w_out + w] = sum;\n");
        code.push_str("        }\n");
    /// Generate standard convolution code
    fn generate_standard_conv(
        h_in: usize,
        w_in: usize,
        code.push_str("  // Standard convolution loops\n");
            "          for (int c_in = 0; c_in < {}; c_in++) {{\n",
            c_in
            "            for (int kh = 0; kh < {}; kh++) {{\n",
            kh
            "              for (int kw = 0; kw < {}; kw++) {{\n",
            kw
            "                int h_in_idx = h * {} - {} + kh;\n",
            stride[0], padding[0]
            "                int w_in_idx = w * {} - {} + kw;\n",
            stride[1], padding[1]
        code.push_str(&format!("                if (h_in_idx >= 0 && h_in_idx < {} && w_in_idx >= 0 && w_in_idx < {}) {{\n", h_in, w_in));
        code.push_str("                  sum += input[((n*c_in + c_in)*h_in + h_in_idx)*w_in + w_in_idx] * weight[((c_out*c_in + c_in)*kh + kh)*kw + kw];\n");
        code.push_str("                }\n");
        code.push_str("              }\n");
        code.push_str("            }\n");
        code.push_str("          }\n");
    /// Generate vectorized activation code
    fn generate_vectorized_activation(
        _total_elements: usize,
        code.push_str(&format!("  // Vectorized activation: {:?}\n", activation));
        code.push_str("    __m256 input_vec = _mm256_loadu_ps(&input[i]);\n");
        match activation {
            ActivationType::ReLU => {
            ActivationType::Sigmoid => {
                code.push_str("    // Sigmoid approximation\n");
                code.push_str("    __m256 one = _mm256_set1_ps(1.0f);\n");
                    "    __m256 neg_input = _mm256_sub_ps(_mm256_setzero_ps(), input_vec);\n",
                    "    // Approximate exp using polynomial (AVX doesn't have native exp)\n",
                    "    __m256 exp_neg = _mm256_add_ps(_mm256_set1_ps(1.0f), neg_input);\n",
                    "    __m256 result = _mm256_div_ps(one_mm256_add_ps(one, exp_neg));\n",
            ActivationType::Tanh => {
                // Tanh approximation: x / (1 + |x|) for simplicity
                    "    __m256 abs_input = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), input_vec);\n",
                code.push_str("    __m256 result = _mm256_div_ps(input_vec_mm256_add_ps(one, abs_input));\n");
                code.push_str("    __m256 result = input_vec; // fallback\n");
        code.push_str("    _mm256_storeu_ps(&output[i], result);\n");
    /// Generate scalar activation code
    fn generate_scalar_activation(
        code.push_str(&format!("  // Scalar activation: {:?}\n", activation));
                code.push_str("    output[i] = fmaxf(0.0f, input[i]);\n");
                code.push_str("    output[i] = 1.0f / (1.0f + expf(-input[i]));\n");
                code.push_str("    output[i] = tanhf(input[i]);\n");
            ActivationType::GELU => {
                code.push_str("    float x = input[i];\n");
                code.push_str("    output[i] = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));\n");
            ActivationType::Swish => {
                code.push_str("    output[i] = x / (1.0f + expf(-x));\n");
            ActivationType::LeakyReLU(alpha) => {
                    "    output[i] = input[i] > 0.0f ? input[i] : {}f * input[i];\n",
                    alpha
                code.push_str("    output[i] = input[i]; // fallback\n");
    /// Analyze memory requirements for an operation
    fn analyze_memory_requirements(&self, operation: &JITOperation) -> Result<MemoryRequirements> {
        let (min_memory, optimal_memory, access_pattern) = match operation {
                ashape, bshape, ..
                let a_size = ashape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let b_size = bshape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let c_size = ashape[0] * bshape[1] * std::mem::size_of::<f32>();
                let min_mem = a_size + b_size + c_size;
                let optimal_mem = min_mem * 2; // For blocking optimizations
                (min_mem, optimal_mem, MemoryAccessPattern::Blocked), JITOperation::ElementWise { shapes, .. } => {
                let total_size = shapes
                    .iter()
                    .map(|shape| shape.iter().product::<usize>() * std::mem::size_of::<f32>())
                    .sum::<usize>();
                (total_size, total_size, MemoryAccessPattern::Sequential)
                ..
                let input_size = inputshape.iter().product::<usize>() * std::mem::size_of::<f32>();
                let weight_size =
                    weightshape.iter().product::<usize>() * std::mem::size_of::<f32>();
                // Output size calculation
                let n = inputshape[0];
                let c_out = weightshape[0];
                let h_out = inputshape[2]; // Simplified
                let w_out = inputshape[3]; // Simplified
                let output_size = n * c_out * h_out * w_out * std::mem::size_of::<f32>();
                let min_mem = input_size + weight_size + output_size;
                let optimal_mem = min_mem + input_size; // For im2col buffer
                (min_mem, optimal_mem, MemoryAccessPattern::Strided)
                // Generic estimation
                (1024, 4096, MemoryAccessPattern::Sequential)
        Ok(MemoryRequirements {
            min_memory,
            optimal_memory,
            access_pattern,
            alignment: 32, // 32-byte alignment for AVX
        })
    /// Estimate performance characteristics
    fn estimate_performance(&self, operation: &JITOperation) -> Result<PerformanceHints> {
        let (flops, memory_bytes, vectorization_factor, parallelization_level) = match operation {
                let m = ashape[0];
                let k = ashape[1];
                let n = bshape[1];
                let flops = 2 * m * k * n; // 2 operations per multiply-add
                let memory_bytes = (m * k + k * n + m * n) * std::mem::size_of::<usize>();
                (flops as u64, memory_bytes, 8, 4) // AVX can process 8 floats, good parallelization
                let elements = shapes[0].iter().product::<usize>();
                let flops = elements; // 1 operation per element
                let memory_bytes = elements * shapes.len() * std::mem::size_of::<usize>();
                (flops as u64, memory_bytes, 8, 8) // Highly parallel
                let c_in = inputshape[1];
                let h_in = inputshape[2];
                let w_in = inputshape[3];
                let kh = weightshape[2];
                let kw = weightshape[3];
                let flops = n * c_out * h_in * w_in * c_in * kh * kw * 2; // Approximate
                let memory_bytes = (inputshape.iter().product::<usize>()
                    + weightshape.iter().product::<usize>())
                    * std::mem::size_of::<usize>();
                (flops as u64, memory_bytes, 4, 4) // Moderate vectorization and parallelization
                (1000, 1024, 1, 1) // Conservative estimates
        let compute_intensity = flops as f64 / memory_bytes as f64;
        let memory_bandwidth_util = (compute_intensity / 100.0).min(1.0); // Heuristic
        Ok(PerformanceHints {
            estimated_flops: flops,
            memory_bandwidth_util,
            compute_intensity,
            vectorization_factor,
            parallelization_level,
    /// Validate inputs for kernel execution
    fn validate_kernel_inputs<F: Float + Debug>(
        _kernel: &CompiledKernel,
    ) -> Result<()> {
        if inputs.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "No inputs provided".to_string(),
        // Additional validation logic would go here
        Ok(())
    /// Execute kernel using compiled native code or optimized fallbacks
    fn execute_kernel_native<F: Float + Debug + 'static>(
        // Execute using compiled function if available
        if let Some(ref compiled_fn) = kernel.compiled_fn {
            return self.execute_compiled_function(compiled_fn, inputs, outputshape);
        // Fallback to optimized SIMD implementations
        self.execute_optimized_fallback(inputs, outputshape)
    /// Execute compiled native function
    fn execute_compiled_function<F: Float + Debug + 'static>(
        compiled_fn: &CompiledFunction,
        match compiled_fn {
            CompiledFunction::ElementWise(kernel) => {
                self.execute_elementwise_kernel(kernel, inputs, outputshape)
            CompiledFunction::MatMul(kernel) => {
                self.execute_matmul_kernel(kernel, inputs, outputshape)
            CompiledFunction::Convolution(kernel) => {
                self.execute_convolution_kernel(kernel, inputs, outputshape)
            CompiledFunction::Activation(kernel) => {
                self.execute_activation_kernel(kernel, inputs, outputshape)
            CompiledFunction::Fused(kernel) => {
                self.execute_fused_kernel(kernel, inputs, outputshape)
    /// Execute element-wise operation kernel with SIMD optimization
    fn execute_elementwise_kernel<F: Float + Debug + 'static>(
        kernel: &ElementWiseKernel,
        if inputs.len() < 1 || inputs.len() > 2 {
                "Element-wise operations support 1-2 inputs".to_string(),
        let mut output = Array::zeros(outputshape).into_dyn();
        // Use f32 optimized path for best performance
        if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() {
            self.execute_elementwise_f32(&kernel.operation, inputs, &mut output)
            self.execute_elementwise_generic(&kernel.operation, inputs, &mut output)
    /// Execute matrix multiplication kernel with optimized blocking
    fn execute_matmul_kernel<F: Float + Debug + 'static>(
        kernel: &MatMulKernel,
        if inputs.len() != 2 {
                "Matrix multiplication requires exactly 2 inputs".to_string(),
            self.execute_matmul_f32_optimized(kernel, inputs, &mut output)
            self.execute_matmul_generic(kernel, inputs, &mut output)
    /// Execute convolution kernel with im2col optimization
    fn execute_convolution_kernel<F: Float + Debug + 'static>(
        kernel: &ConvolutionKernel,
                "Convolution requires exactly 2 inputs (input, weight)".to_string(),
        if kernel.use_im2col {
            self.execute_convolution_im2col(kernel, inputs, &mut output)
            self.execute_convolution_direct(kernel, inputs, &mut output)
    /// Execute activation function kernel with SIMD
    fn execute_activation_kernel<F: Float + Debug + 'static>(
        kernel: &ActivationKernel,
        if inputs.len() != 1 {
                "Activation functions require exactly 1 input".to_string(),
            self.execute_activation_f32_optimized(kernel, inputs, &mut output)
            self.execute_activation_generic(kernel, inputs, &mut output)
    /// Execute fused operation kernel
    fn execute_fused_kernel<F: Float + Debug + 'static>(
        kernel: &FusedKernel,
        match kernel.fusion_strategy {
            FusionStrategy::Vertical => self.execute_vertical_fusion(kernel, inputs, outputshape),
                self.execute_horizontal_fusion(kernel, inputs, outputshape)
                // Fallback to sequential execution
                self.execute_sequential_fusion(kernel, inputs, outputshape)
    /// Optimized fallback implementation using scirs2-core
    fn execute_optimized_fallback<F: Float + Debug + 'static>(
        if inputs.len() == 2 {
            // Binary operation - default to addition with SIMD
            if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() {
                unsafe {
                    let input0 = &*(inputs[0] as *const ArrayD<F> as *const ArrayD<f32>);
                    let input1 = &*(inputs[1] as *const ArrayD<F> as *const ArrayD<f32>);
                    let output_f32 = &mut *(&mut output as *mut ArrayD<F> as *mut ArrayD<f32>);
                    // Use scirs2-core SIMD operations
                    f32::simd_add(&input0.view(), &input1.view(), &mut output_f32.view_mut());
            } else {
                // Generic fallback
                par_azip!((out in &mut output, a in inputs[0], b in inputs[1]) {
                    *out = *a + *b;
                });
        } else if inputs.len() == 1 {
            // Unary operation - default to ReLU
            par_azip!((out in &mut output, inp in inputs[0]) {
                *out = if *inp > F::zero() { *inp } else { F::zero() };
            });
    /// Update cache statistics
    fn update_cache_stats(&self, cachehit: bool) {
        if let Ok(mut stats) = self.stats.write() {
            let total_requests = stats.kernels_compiled + if cache_hit { 1 } else { 0 };
            if total_requests > 0 {
                let hits = if cache_hit {
                    (stats.cache_hit_rate * (total_requests - 1) as f64) + 1.0
                    stats.cache_hit_rate * (total_requests - 1) as f64
                };
                stats.cache_hit_rate = hits / total_requests as f64;
    /// Update compilation statistics
    fn update_compile_stats(&self, compile_timems: f64) {
            stats.kernels_compiled += 1;
            stats.total_compile_time_ms += compile_time_ms;
            stats.avg_compile_time_ms = stats.total_compile_time_ms / stats.kernels_compiled as f64;
    /// Update execution statistics
    fn update_execution_stats(&self, execution_timems: f64) {
            stats.total_execution_time_ms += execution_time_ms;
    /// Get compilation and execution statistics
    pub fn get_statistics(&self) -> JITStatistics {
        if let Ok(stats) = self.stats.read() {
            stats.clone()
            JITStatistics::default()
    /// Clear the kernel cache
    pub fn clear_cache(&self) {
            cache.clear();
    /// Get cache size
    pub fn cache_size(&self) -> usize {
            cache.len()
            0
    /// Set optimization level
    pub fn set_optimization_level(&mut self, level: OptimizationLevel) {
        self.optimization_level = level;
    /// Set code generation settings
    pub fn set_codegen_settings(&mut self, settings: CodeGenSettings) {
        self.codegen_settings = settings;
    /// Compile operation to native executable code
    fn compile_to_native(&self, operation: &JITOperation) -> Result<CompiledFunction> {
            JITOperation::ElementWise { op, shapes } => {
                let kernel = ElementWiseKernel {
                    operation: op.clone(),
                    input_count: shapes.len(),
                    use_simd: self.target_arch_supports_simd(),
                Ok(CompiledFunction::ElementWise(kernel))
                let m = if *transpose_a { ashape[1] } else { ashape[0] };
                let k = if *transpose_a { ashape[0] } else { ashape[1] };
                let n = if *transpose_b { bshape[0] } else { bshape[1] };
                let kernel = MatMulKernel {
                    m,
                    k,
                    n,
                    transpose_a: *transpose_a,
                    transpose_b: *transpose_b,
                    block_size: self.calculate_optimal_block_size(m, k, n),
                Ok(CompiledFunction::MatMul(kernel))
                let kernel = ConvolutionKernel {
                    inputshape: inputshape.clone(),
                    weightshape: weightshape.clone(),
                    stride: stride.clone(),
                    padding: padding.clone(),
                    use_im2col: self.should_use_im2col(inputshape, weightshape),
                Ok(CompiledFunction::Convolution(kernel)), JITOperation::Activation { activation, .. } => {
                let kernel = ActivationKernel {
                    activation: activation.clone(),
                    use_lookup_table: self.should_use_lookup_table(activation),
                Ok(CompiledFunction::Activation(kernel))
                let mut compiled_ops = Vec::new();
                for op in operations {
                    compiled_ops.push(self.compile_to_native(op)?);
                let kernel = FusedKernel {
                    operations: compiled_ops,
                    fusion_strategy: fusion_strategy.clone(),
                    buffer_reuse: true,
                Ok(CompiledFunction::Fused(kernel))
            _ => Err(NeuralError::ComputationError(
                "Operation type not yet supported for native compilation".to_string(),
            )),
    /// Generate optimization metadata for a kernel
    fn generate_optimization_metadata(
    ) -> Result<OptimizationMetadata> {
        let simd_level = match &self.target_arch {
            TargetArchitecture::X86_64 { avx_level, .. } => match *avx_level {
                512 => SimdLevel::AVX512,
                2 => SimdLevel::AVX2,
                1 => SimdLevel::AVX_ => SimdLevel::SSE,
            },
            TargetArchitecture::ARM64 { neon: true, .. } => SimdLevel::NEON_ => SimdLevel::None,
        let memory_optimization = match operation {
            JITOperation::MatMul { .. } => MemoryOptimization::Blocking,
            JITOperation::Convolution { .. } => MemoryOptimization::TileOptimized,
            JITOperation::ElementWise { .. } => MemoryOptimization::Streaming_ => MemoryOptimization::None,
        let parallel_strategy = if self.codegen_settings.use_intrinsics {
            ParallelStrategy::Hybrid
            ParallelStrategy::Rayon
        Ok(OptimizationMetadata {
            simd_level,
            memory_optimization,
            unroll_factor: if self.codegen_settings.unroll_loops {
                4
                1
            parallel_strategy,
            cache_blocks: self.calculate_cache_blocks(operation),
    /// Calculate optimal block size for matrix operations
    fn calculate_optimal_block_size(&self, m: usize, k: usize, n: usize) -> usize {
        // Use L1 cache size heuristic: aim for blocks that fit in L1 cache
        let l1_cache_size = 32 * 1024; // 32KB typical L1 cache
        let element_size = std::mem::size_of::<f32>();
        let target_elements = l1_cache_size / (3 * element_size); // A, B, C blocks
        // Find block size that divides dimensions well
        let max_dim = m.max(k).max(n);
        let mut block_size = (target_elements as f64).sqrt() as usize;
        // Round to nearest power of 2 or multiple of 8 for SIMD alignment
        block_size = (block_size / 8) * 8;
        block_size.max(8).min(max_dim / 4)
    /// Determine if im2col should be used for convolution
    fn should_use_im2col(&self, inputshape: &[usize], weightshape: &[usize]) -> bool {
        if inputshape.len() < 4 || weightshape.len() < 4 {
            return false;
        let kh = weightshape[2];
        let kw = weightshape[3];
        let c_in = weightshape[1];
        // Use im2col for larger kernels or many input channels
        kh * kw * c_in > 64
    /// Determine if lookup table should be used for activation
    fn should_use_lookup_table(&self, activation: &ActivationType) -> bool {
            ActivationType::Sigmoid | ActivationType::Tanh => {
                self.codegen_settings.use_lookup_tables
    /// Calculate cache-friendly block sizes
    fn calculate_cache_blocks(&self, operation: &JITOperation) -> Vec<usize> {
            JITOperation::MatMul { ashape, .. } => {
                let block_size =
                    self.calculate_optimal_block_size(ashape[0], ashape[1], ashape[1]);
                vec![block_size, block_size]
            JITOperation::Convolution { inputshape, .. } => {
                if inputshape.len() >= 4 {
                    let h = inputshape[2];
                    let w = inputshape[3];
                    vec![(h / 4).max(1), (w / 4).max(1)]
                    vec![1]
            _ => vec![1],
impl FusionOptimizer {
    /// Create a new fusion optimizer with enhanced rules
    pub fn new() -> Self {
        let mut fusion_rules = HashMap::new();
        // Enhanced fusion rules for better performance
        fusion_rules.insert(
            ("elementwise".to_string(), "elementwise".to_string()),
            FusionRule {
                can_fuse: true,
                performance_gain: 1.8, // Higher gain with SIMD fusion
                memory_savings: 0,
                strategy: FusionStrategy::Horizontal,
            ("activation".to_string(), "elementwise".to_string()),
                performance_gain: 1.5, // Better with kernel fusion
                strategy: FusionStrategy::Vertical,
            ("matmul".to_string(), "activation".to_string()),
                performance_gain: 1.3,
            ("convolution".to_string(), "activation".to_string()),
                performance_gain: 1.4,
            ("convolution".to_string(), "normalization".to_string()),
                performance_gain: 1.25,
            fusion_rules,
            max_fusion_depth: 6, // Increased for better optimization
            memory_threshold: 2 * 1024 * 1024, // 2MB for larger buffers
    /// Optimize an operation by applying fusion rules
    pub fn optimize_operation(&self, operation: &JITOperation) -> Result<JITOperation> {
        // Apply operation-specific optimizations
            JITOperation::FusedOp { operations, .. } => {
                // Already fused, apply further optimizations
                self.optimize_fused_operation(operation)
                // Look for fusion opportunities with common patterns
                self.apply_single_operation_optimizations(operation)
    /// Optimize already fused operations
    fn optimize_fused_operation(&self, operation: &JITOperation) -> Result<JITOperation> {
        if let JITOperation::FusedOp {
            operations,
            fusion_strategy,
        } = operation
            // Reorder operations for better cache locality
            let optimized_ops = self.reorder_for_cache_efficiency(operations)?;
            // Apply memory layout optimizations
            let strategy = self.select_optimal_fusion_strategy(&optimized_ops, fusion_strategy)?;
            Ok(JITOperation::FusedOp {
                operations: optimized_ops,
                fusion_strategy: strategy,
            })
            Ok(operation.clone())
    /// Apply optimizations to single operations
    fn apply_single_operation_optimizations(
    ) -> Result<JITOperation> {
                // Optimize element-wise operations for SIMD
                Ok(JITOperation::ElementWise {
                    op: self.optimize_elementwise_op(op)?,
                    shapes: self.optimizeshapes_for_simd(shapes),
                })
                // Apply matrix multiplication optimizations
                let (opt_ashape, opt_bshape, opt_transpose_a, opt_transpose_b) =
                    self.optimize_matmul_layout(ashape, bshape, *transpose_a, *transpose_b)?;
                Ok(JITOperation::MatMul {
                    ashape: opt_ashape,
                    bshape: opt_bshape,
                    transpose_a: opt_transpose_a,
                    transpose_b: opt_transpose_b,
                // Optimize convolution parameters
                Ok(JITOperation::Convolution {
                    inputshape: self.optimize_conv_input_layout(inputshape),
                    weightshape: self.optimize_conv_weight_layout(weightshape),
                    dilation: dilation.clone(, _ => Ok(operation.clone()),
    /// Reorder operations for better cache efficiency
    fn reorder_for_cache_efficiency(
    ) -> Result<Vec<Box<JITOperation>>> {
        let mut ops = operations.to_vec();
        // Sort by memory access patterns: sequential first, then blocked
        ops.sort_by(|a, b| {
            let a_score = self.get_memory_locality_score(a);
            let b_score = self.get_memory_locality_score(b);
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(ops)
    /// Get memory locality score for operation ordering
    fn get_memory_locality_score(&self, operation: &JITOperation) -> f64 {
            JITOperation::ElementWise { .. } => 1.0, // Best locality
            JITOperation::Activation { .. } => 0.9,
            JITOperation::MatMul { .. } => 0.7,
            JITOperation::Convolution { .. } => 0.5_ => 0.0,
    /// Select optimal fusion strategy
    fn select_optimal_fusion_strategy(
        current_strategy: &FusionStrategy,
    ) -> Result<FusionStrategy> {
        if operations.len() <= 1 {
            return Ok(current_strategy.clone());
        // Analyze data dependencies
        let has_dependencies = self.analyze_data_dependencies(operations);
        if has_dependencies {
            Ok(FusionStrategy::Vertical)
            Ok(FusionStrategy::Horizontal)
    /// Analyze data dependencies between operations
    fn analyze_data_dependencies(&self, operations: &[Box<JITOperation>]) -> bool {
        // Simple heuristic: if we have different operation types, assume dependencies
        let mut op_types = std::collections::HashSet::new();
        for op in operations {
            op_types.insert(self.operation_type(op));
        op_types.len() > 1
    /// Optimize element-wise operation
    fn optimize_elementwise_op(&self, op: &ElementWiseOp) -> Result<ElementWiseOp> {
                // Try to simplify custom expressions
                Ok(ElementWiseOp::Custom(self.simplify_expression(expr)))
            _ => Ok(op.clone()),
    /// Optimize shapes for SIMD alignment
    fn optimizeshapes_for_simd(&self, shapes: &[Vec<usize>]) -> Vec<Vec<usize>> {
        shapes
            .iter()
            .map(|shape| {
                let mut optshape = shape.clone();
                if let Some(last) = optshape.last_mut() {
                    // Pad to SIMD-friendly size (multiple of 8 for AVX)
                    *last = (*last + 7) / 8 * 8;
                optshape
            .collect()
    /// Optimize matrix multiplication layout
    fn optimize_matmul_layout(
    ) -> Result<(Vec<usize>, Vec<usize>, bool, bool)> {
        // Heuristic: prefer column-major B matrix for better cache performance
        let opt_transpose_b = if !transpose_b && bshape[0] > bshape[1] {
            true
            transpose_b
        Ok((
            ashape.to_vec(),
            bshape.to_vec(),
            transpose_a,
            opt_transpose_b,
        ))
    /// Optimize convolution input layout
    fn optimize_conv_input_layout(&self, inputshape: &[usize]) -> Vec<usize> {
        let mut optshape = inputshape.to_vec();
        // Ensure channel dimension is SIMD-friendly
        if optshape.len() >= 2 {
            optshape[1] = (optshape[1] + 7) / 8 * 8;
        optshape
    /// Optimize convolution weight layout
    fn optimize_conv_weight_layout(&self, weightshape: &[usize]) -> Vec<usize> {
        let mut optshape = weightshape.to_vec();
        // Ensure output channels are SIMD-friendly
        if !optshape.is_empty() {
            optshape[0] = (optshape[0] + 7) / 8 * 8;
    /// Simplify mathematical expressions
    fn simplify_expression(&self, expr: &str) -> String {
        // Basic expression simplification
        expr.replace(" ", "")
            .replace("+-", "-")
            .replace("-+", "-")
            .replace("++", "+")
            .replace("--", "+")
    /// Check if two operations can be fused with enhanced analysis
    pub fn can_fuse(&self, op1: &JITOperation, op2: &JITOperation) -> bool {
        let key = (self.operation_type(op1), self.operation_type(op2));
        if let Some(rule) = self.fusion_rules.get(&key) {
            if !rule.can_fuse {
                return false;
            // Additional checks for fusion compatibility
            self.check_memory_compatibility(op1, op2)
                && self.checkshape_compatibility(op1, op2)
                && self.check_compute_intensity_compatibility(op1, op2)
            false
    /// Check memory access pattern compatibility
    fn check_memory_compatibility(&self, op1: &JITOperation, op2: &JITOperation) -> bool {
        let pattern1 = self.get_memory_access_pattern(op1);
        let pattern2 = self.get_memory_access_pattern(op2);
        // Compatible patterns can be fused
        match (pattern1, pattern2) {
            (MemoryAccessPattern::Sequential, MemoryAccessPattern::Sequential) => true,
            (MemoryAccessPattern::Blocked, MemoryAccessPattern::Blocked) => true,
            (MemoryAccessPattern::Sequential, MemoryAccessPattern::Strided) => true,
    /// Check shape compatibility for fusion
    fn checkshape_compatibility(&self, op1: &JITOperation, op2: &JITOperation) -> bool {
        let shapes1 = self.get_operationshapes(op1);
        let shapes2 = self.get_operationshapes(op2);
        // Must have compatible output/input shapes
        if shapes1.is_empty() || shapes2.is_empty() {
        // Check if output of op1 can be input to op2
        shapes1.last() == shapes2.first()
    /// Check compute intensity compatibility
    fn check_compute_intensity_compatibility(
        op1: &JITOperation,
        op2: &JITOperation,
    ) -> bool {
        let intensity1 = self.estimate_compute_intensity(op1);
        let intensity2 = self.estimate_compute_intensity(op2);
        // Fuse operations with similar compute intensity
        let ratio = intensity1.max(intensity2) / intensity1.min(intensity2).max(1e-6);
        ratio < 10.0 // Allow up to 10x difference
    /// Get memory access pattern for operation
    fn get_memory_access_pattern(&self, operation: &JITOperation) -> MemoryAccessPattern {
            JITOperation::ElementWise { .. } => MemoryAccessPattern::Sequential,
            JITOperation::MatMul { .. } => MemoryAccessPattern::Blocked,
            JITOperation::Convolution { .. } => MemoryAccessPattern::Strided,
            JITOperation::Activation { .. } => MemoryAccessPattern::Sequential_ => MemoryAccessPattern::Random,
    /// Get shapes involved in operation
    fn get_operationshapes(&self, operation: &JITOperation) -> Vec<Vec<usize>> {
            JITOperation::ElementWise { shapes, .. } => shapes.clone(),
                vec![ashape.clone(), bshape.clone()]
                vec![inputshape.clone(), weightshape.clone()]
            JITOperation::Activation { inputshape, .. } => {
                vec![inputshape.clone()]
            _ => Vec::new(),
    /// Estimate compute intensity (FLOPS per byte)
    fn estimate_compute_intensity(&self, operation: &JITOperation) -> f64 {
            JITOperation::ElementWise { .. } => 0.25, // 1 op per 4 bytes
                let m = ashape[0] as f64;
                let k = ashape[1] as f64;
                let n = bshape[1] as f64;
                let flops = 2.0 * m * k * n;
                let bytes = (m * k + k * n + m * n) * 4.0; // f32
                flops / bytes
                if inputshape.len() >= 4 && weightshape.len() >= 4 {
                    let input_size = inputshape.iter().product::<usize>() as f64;
                    let weight_size = weightshape.iter().product::<usize>() as f64;
                    let flops = input_size * weight_size.sqrt(); // Approximation
                    let bytes = (input_size + weight_size) * 4.0;
                    flops / bytes
                    1.0
            JITOperation::Activation { .. } => 0.5, // 2 ops per 4 bytes
            _ => 1.0,
    /// Get operation type string for fusion rules
    fn operation_type(&self, operation: &JITOperation) -> String {
            JITOperation::MatMul { .. } => "matmul".to_string(),
            JITOperation::ElementWise { .. } => "elementwise".to_string(),
            JITOperation::Convolution { .. } => "convolution".to_string(),
            JITOperation::Activation { .. } => "activation".to_string(),
            JITOperation::Pooling { .. } => "pooling".to_string(),
            JITOperation::Normalization { .. } => "normalization".to_string(),
            JITOperation::Reduction { .. } => "reduction".to_string(),
            JITOperation::FusedOp { .. } => "fused".to_string(),
    /// Get estimated performance gain for fusion
    pub fn get_fusion_performance_gain(&self, op1: &JITOperation, op2: &JITOperation) -> f64 {
        self.fusion_rules
            .get(&key)
            .map_or(1.0, |rule| rule.performance_gain)
    /// Get estimated memory savings for fusion
    pub fn get_fusion_memory_savings(&self, op1: &JITOperation, op2: &JITOperation) -> usize {
            // Calculate intermediate buffer size that would be saved
            let shapes1 = self.get_operationshapes(op1);
            if let Some(outputshape) = shapes1.last() {
                let intermediate_size =
                    outputshape.iter().product::<usize>() * std::mem::size_of::<f32>();
                rule.memory_savings + intermediate_size
                rule.memory_savings
    /// Execute f32 element-wise operations with SIMD optimization
    fn execute_elementwise_f32<F: Float + Debug + 'static>(
        operation: &ElementWiseOp,
        output: &mut ArrayD<F>,
        if std::any::TypeId::of::<F>() != std::any::TypeId::of::<f32>() {
            return self.execute_elementwise_generic(operation, inputs, output);
        unsafe {
            // Cast to f32 for SIMD operations
            let input0_f32 = &*(inputs[0] as *const ArrayD<F> as *const ArrayD<f32>);
            let output_f32 = &mut *(output as *mut ArrayD<F> as *mut ArrayD<f32>);
            match operation {
                ElementWiseOp::Add if inputs.len() == 2 => {
                    let input1_f32 = &*(inputs[1] as *const ArrayD<F> as *const ArrayD<f32>);
                    f32::simd_add(
                        &input0_f32.view(),
                        &input1_f32.view(),
                        &mut output_f32.view_mut(),
                    );
                ElementWiseOp::Mul if inputs.len() == 2 => {
                    f32::simd_mul(
                ElementWiseOp::Sub if inputs.len() == 2 => {
                    f32::simd_sub(
                ElementWiseOp::Abs => {
                    f32::simd_abs(&input0_f32.view(), &mut output_f32.view_mut());
                ElementWiseOp::Sqrt => {
                    f32::simd_sqrt(&input0_f32.view(), &mut output_f32.view_mut());
                ElementWiseOp::Exp => {
                    f32::simd_exp(&input0_f32.view(), &mut output_f32.view_mut());
                ElementWiseOp::Log => {
                    f32::simd_log(&input0_f32.view(), &mut output_f32.view_mut());
                _ => {
                    // Fallback to generic implementation
                    return self.execute_elementwise_generic(operation, inputs, output);
    /// Execute generic element-wise operations
    fn execute_elementwise_generic<F: Float + Debug + 'static>(
            ElementWiseOp::Add if inputs.len() == 2 => {
                par_azip!((out in output, a in inputs[0], b in inputs[1]) {
            ElementWiseOp::Sub if inputs.len() == 2 => {
                    *out = *a - *b;
            ElementWiseOp::Mul if inputs.len() == 2 => {
                    *out = *a * *b;
            ElementWiseOp::Div if inputs.len() == 2 => {
                    *out = *a / *b;
                par_azip!((out in output, inp in inputs[0]) {
                    *out = inp.abs();
            ElementWiseOp::Sqrt => {
                    *out = inp.sqrt();
            ElementWiseOp::Exp => {
                    *out = inp.exp();
            ElementWiseOp::Log => {
                    *out = inp.ln();
            ElementWiseOp::Sin => {
                    *out = inp.sin();
            ElementWiseOp::Cos => {
                    *out = inp.cos();
            ElementWiseOp::Tanh => {
                    *out = inp.tanh();
                return Err(NeuralError::ComputationError(format!(
                    "Unsupported element-wise operation: {:?}",
                    operation
                )));
    /// Execute optimized f32 matrix multiplication
    fn execute_matmul_f32_optimized<F: Float + Debug + 'static>(
            return self.execute_matmul_generic(kernel, inputs, output);
            let a_f32 = &*(inputs[0] as *const ArrayD<F> as *const ArrayD<f32>);
            let b_f32 = &*(inputs[1] as *const ArrayD<F> as *const ArrayD<f32>);
            let c_f32 = &mut *(output as *mut ArrayD<F> as *mut ArrayD<f32>);
            // Use blocked matrix multiplication for better cache performance
            self.blocked_matmul_f32(
                &a_f32.view(),
                &b_f32.view(),
                &mut c_f32.view_mut(),
                kernel.m,
                kernel.k,
                kernel.n,
                kernel.block_size,
                kernel.transpose_a,
                kernel.transpose_b,
            )?;
    /// Execute generic matrix multiplication
    fn execute_matmul_generic<F: Float + Debug + 'static>(
        let a = inputs[0];
        let b = inputs[1];
        // Simple parallel matrix multiplication
        par_azip!((index (i, j), out in output) {
            let mut sum = F::zero();
            for k in 0..kernel.k {
                let a_val = if kernel.transpose_a {
                    a[[k, i]]
                    a[[i, k]]
                let b_val = if kernel.transpose_b {
                    b[[j, k]]
                    b[[k, j]]
                sum = sum + a_val * b_val;
            *out = sum;
    /// Blocked matrix multiplication for f32
    fn blocked_matmul_f32(
        a: &ArrayView<f32, scirs2_core::ndarray::Ix2>,
        b: &ArrayView<f32, scirs2_core::ndarray::Ix2>,
        c: &mut ArrayViewMut<f32, scirs2_core::ndarray::Ix2>,
        block_size: usize,
        // Blocked algorithm for better cache performance
        par_azip!((index (bi, bj)_c in c.blocks(block_size, block_size)) {
            for bk in (0..k).step_by(block_size) {
                let m_end = (bi + block_size).min(m);
                let n_end = (bj + block_size).min(n);
                let k_end = (bk + block_size).min(k);
                for i in bi..m_end {
                    for j in bj..n_end {
                        let mut sum = if bk == 0 { 0.0 } else { c[[i, j]] };
                        for kk in bk..k_end {
                            let a_val = if transpose_a {
                                a[[kk, i]]
                            } else {
                                a[[i, kk]]
                            };
                            let b_val = if transpose_b {
                                b[[j, kk]]
                                b[[kk, j]]
                            sum += a_val * b_val;
                        }
                        c[[i, j]] = sum;
                    }
    /// Execute convolution with im2col
    fn execute_convolution_im2col<F: Float + Debug + 'static>(
        // Im2col-based convolution for better performance with larger kernels
        // This transforms convolution into a matrix multiplication
        let input = inputs[0];
        let weight = inputs[1];
        if input.ndim() != 4 || weight.ndim() != 4 {
            return Err(NeuralError::ComputationError(
                "Convolution requires 4D tensors".to_string(),
        let n = input.shape()[0];
        let c_in = input.shape()[1];
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];
        let c_out = weight.shape()[0];
        let kh = weight.shape()[2];
        let kw = weight.shape()[3];
        let h_out = (h_in + 2 * kernel.padding[0] - kh) / kernel.stride[0] + 1;
        let w_out = (w_in + 2 * kernel.padding[1] - kw) / kernel.stride[1] + 1;
        // Parallel over batch and output spatial dimensions
        par_azip!((index (batch, out_h, out_w)_out in output.slice_mut(s![.., .., ..h_out, ..w_out])) {
            for c_o in 0..c_out {
                let mut sum = F::zero();
                for c_i in 0..c_in {
                    for kh_i in 0..kh {
                        for kw_i in 0..kw {
                            let h_idx = out_h * kernel.stride[0] + kh_i;
                            let w_idx = out_w * kernel.stride[1] + kw_i;
                            if h_idx >= kernel.padding[0] && h_idx < h_in + kernel.padding[0] &&
                               w_idx >= kernel.padding[1] && w_idx < w_in + kernel.padding[1] {
                                let input_h = h_idx - kernel.padding[0];
                                let input_w = w_idx - kernel.padding[1];
                                if input_h < h_in && input_w < w_in {
                                    let input_val = input[[batch, c_i, input_h, input_w]];
                                    let weight_val = weight[[c_o, c_i, kh_i, kw_i]];
                                    sum = sum + input_val * weight_val;
                                }
                            }
                output[[batch, c_o, out_h, out_w]] = sum;
    /// Execute direct convolution
    fn execute_convolution_direct<F: Float + Debug + 'static>(
        // Direct convolution implementation
        self.execute_convolution_im2col(kernel, inputs, output)
    /// Execute optimized f32 activation functions
    fn execute_activation_f32_optimized<F: Float + Debug + 'static>(
            return self.execute_activation_generic(kernel, inputs, output);
            let input_f32 = &*(inputs[0] as *const ArrayD<F> as *const ArrayD<f32>);
            match kernel.activation {
                ActivationType::ReLU => {
                    f32::simd_relu(&input_f32.view(), &mut output_f32.view_mut());
                ActivationType::Sigmoid => {
                    f32::simd_sigmoid(&input_f32.view(), &mut output_f32.view_mut());
                ActivationType::Tanh => {
                    f32::simd_tanh(&input_f32.view(), &mut output_f32.view_mut());
                ActivationType::GELU => {
                    f32::simd_gelu(&input_f32.view(), &mut output_f32.view_mut());
                ActivationType::Swish => {
                    f32::simd_swish(&input_f32.view(), &mut output_f32.view_mut());
                    return self.execute_activation_generic(kernel, inputs, output);
    /// Execute generic activation functions
    fn execute_activation_generic<F: Float + Debug + 'static>(
        match kernel.activation {
                par_azip!((out in output, inp in input) {
                    *out = if *inp > F::zero() { *inp } else { F::zero() };
                    *out = F::one() / (F::one() + (-*inp).exp());
                    let x = *inp;
                    let sqrt_2_pi = F::from(0.797885).unwrap();
                    let coeff = F::from(0.044715).unwrap();
                    let inner = sqrt_2_pi * (x + coeff * x * x * x);
                    *out = F::from(0.5).unwrap() * x * (F::one() + inner.tanh());
                    *out = x / (F::one() + (-x).exp());
                let alpha_f = F::from(*alpha).unwrap();
                    *out = if *inp > F::zero() { *inp } else { alpha_f * *inp };
                    "Unsupported activation function: {:?}",
                    kernel.activation
    /// Execute vertical fusion (pipeline operations)
    fn execute_vertical_fusion<F: Float + Debug + 'static>(
        let mut current_output = inputs[0].to_owned();
        for op in &kernel.operations {
            let temp_inputs = vec![&current_output];
            let tempshape = current_output.shape().to_vec();
            current_output = self.execute_compiled_function(op, &temp_inputs, &tempshape)?;
        // Reshape to target output if needed
        if current_output.shape() != outputshape {
            current_output = current_output
                .into_shape(outputshape)
                .map_err(|e| NeuralError::ComputationError(format!("Shape error: {}", e)))?;
        Ok(current_output)
    /// Execute horizontal fusion (parallel operations)
    fn execute_horizontal_fusion<F: Float + Debug + 'static>(
        // Execute operations in parallel and combine results
        let results: Result<Vec<_>> = kernel
            .operations
            .par_iter()
            .enumerate()
            .map(|(i, op)| {
                let op_inputs = if i < inputs.len() {
                    vec![inputs[i]]
                    vec![inputs[0]]
                let tempshape = op_inputs[0].shape().to_vec();
                self.execute_compiled_function(op, &op_inputs, &tempshape)
            .collect();
        let results = results?;
        if results.is_empty() {
            return Ok(Array::zeros(outputshape).into_dyn());
        // Combine results by element-wise addition
        let mut combined = results[0].clone();
        for result in results.iter().skip(1) {
            par_azip!((out in &mut combined, inp in result) {
                *out = *out + *inp;
        Ok(combined)
    /// Execute sequential fusion (fallback)
    fn execute_sequential_fusion<F: Float + Debug + 'static>(
        // Sequential execution of fused operations
        self.execute_vertical_fusion(kernel, inputs, outputshape)
/// Represents a fusion opportunity between operations
pub struct FusionOpportunity {
    /// Index of first operation
    pub op1_index: usize,
    /// Index of second operation  
    pub op2_index: usize,
    /// Expected performance gain ratio
    /// Memory savings in bytes
    /// Complexity of fusion (distance between operations)
    pub fusion_complexity: usize,
impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
impl Default for CodeGenSettings {
        let mut target_features = HashSet::new();
        // Add platform-specific features
            if is_x86_feature_detected!("avx2") {
                target_features.insert("avx2".to_string());
            if is_x86_feature_detected!("fma") {
                target_features.insert("fma".to_string());
            target_features.insert("neon".to_string());
            target_features,
/// Detect AVX support level on x86_64
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
fn detect_avx_level() -> u8 {
    if is_x86_feature_detected!("avx512f") {
        return 512;
    if is_x86_feature_detected!("avx2") {
        return 2;
    if is_x86_feature_detected!("avx") {
        return 1;
    0
/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_jit_compiler_creation() {
        let target_arch = JITCompiler::detect_target_architecture();
        let compiler = JITCompiler::new(target_arch);
        assert_eq!(compiler.cache_size(), 0);
    fn test_matrix_multiplication_compilation() {
        let target_arch = TargetArchitecture::Generic;
        let operation = JITOperation::MatMul {
            ashape: vec![16, 32],
            bshape: vec![32, 64],
            transpose_a: false,
            transpose_b: false,
        let result = compiler.compile_operation(&operation);
        assert!(result.is_ok());
        let kernel = result.unwrap();
        assert!(kernel.code.contains("matmul"));
        assert!(kernel.entry_point.starts_with("kernel_"));
    fn test_element_wise_compilation() {
        let operation = JITOperation::ElementWise {
            op: ElementWiseOp::Add,
            shapes: vec![vec![1024, 512], vec![1024, 512]],
        assert!(kernel.code.contains("elementwise"));
    fn test_convolution_compilation() {
        let operation = JITOperation::Convolution {
            inputshape: vec![1, 3, 224, 224],
            weightshape: vec![64, 3, 7, 7],
            stride: vec![2, 2],
            padding: vec![3, 3],
            dilation: vec![1, 1],
        assert!(kernel.code.contains("conv"));
    fn test_activation_compilation() {
        let operation = JITOperation::Activation {
            activation: ActivationType::ReLU,
            inputshape: vec![32, 128, 56, 56],
        assert!(kernel.code.contains("activation"));
    fn test_cache_functionality() {
            op: ElementWiseOp::Mul,
            shapes: vec![vec![100, 100]],
        // First compilation
        let result1 = compiler.compile_operation(&operation);
        assert!(result1.is_ok());
        assert_eq!(compiler.cache_size(), 1);
        // Second compilation should hit cache
        let result2 = compiler.compile_operation(&operation);
        assert!(result2.is_ok());
    fn test_fusion_optimizer() {
        let optimizer = FusionOptimizer::new();
        let op1 = JITOperation::ElementWise {
        let op2 = JITOperation::ElementWise {
        assert!(optimizer.can_fuse(&op1, &op2));
    fn test_memory_requirements_analysis() {
            ashape: vec![100, 200],
            bshape: vec![200, 300],
        let requirements = compiler.analyze_memory_requirements(&operation);
        assert!(requirements.is_ok());
        let mem_req = requirements.unwrap();
        assert!(mem_req.min_memory > 0);
        assert!(mem_req.optimal_memory >= mem_req.min_memory);
    fn test_performance_estimation() {
        let performance = compiler.estimate_performance(&operation);
        assert!(performance.is_ok());
        let perf_hints = performance.unwrap();
        assert!(perf_hints.estimated_flops > 0);
        assert!(perf_hints.compute_intensity >= 0.0);
    fn test_target_architecture_detection() {
        // Should detect some valid architecture
        match target_arch {
            TargetArchitecture::X86_64 { .. }
            | TargetArchitecture::ARM64 { .. }
            | TargetArchitecture::RISCV { .. }
            | TargetArchitecture::GPU { .. }
            | TargetArchitecture::WASM { .. }
            | TargetArchitecture::Generic => {
                // All valid architectures
    fn test_statistics_tracking() {
            shapes: vec![vec![10, 10]],
        // Compile operation
        let _result = compiler.compile_operation(&operation);
        let stats = compiler.get_statistics();
        assert_eq!(stats.kernels_compiled, 1);
        assert!(stats.total_compile_time_ms >= 0.0);
    fn test_code_generation_settings() {
        let mut compiler = JITCompiler::new(target_arch);
        let mut settings = CodeGenSettings::default();
        settings.vectorize = false;
        settings.unroll_loops = false;
        compiler.set_codegen_settings(settings);
    fn test_optimization_levels() {
        // Test different optimization levels
        let levels = vec![
            OptimizationLevel::O0,
            OptimizationLevel::O1,
            OptimizationLevel::O2,
            OptimizationLevel::O3,
            OptimizationLevel::Os,
        ];
            shapes: vec![vec![50, 50]],
        for level in levels {
            compiler.set_optimization_level(level);
            let result = compiler.compile_operation(&operation);
            assert!(result.is_ok());
