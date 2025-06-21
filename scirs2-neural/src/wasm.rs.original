//! WebAssembly target support for neural networks
//!
//! This module provides comprehensive WebAssembly compilation and deployment support including:
//! - WASM module generation with optimized neural network execution
//! - JavaScript/TypeScript bindings for web integration
//! - WebGL/WebGPU acceleration support
//! - Memory management and streaming for large models
//! - Web Workers integration for background inference
//! - Progressive loading and caching strategies

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::PackageMetadata;
use num_traits::Float;
// HashMap import removed as unused
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};
// Serde imports removed as unused

/// WebAssembly compilation configuration
#[derive(Debug, Clone)]
pub struct WasmCompilationConfig {
    /// Target WASM version
    pub target_version: WasmVersion,
    /// Optimization level
    pub optimization_level: WasmOptimization,
    /// Feature enablement
    pub features: WasmFeatures,
    /// Memory configuration
    pub memory_config: WasmMemoryConfig,
    /// Export configuration
    pub exports: WasmExports,
    /// Import configuration
    pub imports: WasmImports,
    /// Debugging options
    pub debug_config: WasmDebugConfig,
}

/// WebAssembly version target
#[derive(Debug, Clone, PartialEq)]
pub enum WasmVersion {
    /// MVP (Minimum Viable Product) - maximum compatibility
    MVP,
    /// WASM with SIMD support
    SIMD,
    /// WASM with multi-threading support
    Threads,
    /// WASM with SIMD and threading
    SIMDThreads,
    /// Future WASM 2.0
    V2_0,
}

/// WebAssembly optimization configuration
#[derive(Debug, Clone)]
pub struct WasmOptimization {
    /// Size optimization level (0-3)
    pub size_level: u8,
    /// Speed optimization level (0-3)
    pub speed_level: u8,
    /// Enable link-time optimization
    pub lto: bool,
    /// Strip debug information
    pub strip_debug: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Function inlining level
    pub inline_level: InlineLevel,
}

/// Function inlining level
#[derive(Debug, Clone, PartialEq)]
pub enum InlineLevel {
    /// No inlining
    None,
    /// Conservative inlining
    Conservative,
    /// Aggressive inlining
    Aggressive,
    /// Profile-guided inlining
    ProfileGuided,
}

/// WebAssembly feature support
#[derive(Debug, Clone)]
pub struct WasmFeatures {
    /// SIMD (Single Instruction, Multiple Data) support
    pub simd: bool,
    /// Multi-threading support
    pub threads: bool,
    /// Bulk memory operations
    pub bulk_memory: bool,
    /// Reference types
    pub reference_types: bool,
    /// Exception handling
    pub exception_handling: bool,
    /// Tail calls
    pub tail_calls: bool,
    /// Multi-value returns
    pub multi_value: bool,
    /// WebAssembly System Interface (WASI)
    pub wasi: bool,
}

/// WebAssembly memory configuration
#[derive(Debug, Clone)]
pub struct WasmMemoryConfig {
    /// Initial memory pages (64KB each)
    pub initial_pages: u32,
    /// Maximum memory pages
    pub maximum_pages: Option<u32>,
    /// Shared memory (for threading)
    pub shared: bool,
    /// Memory growth strategy
    pub growth_strategy: MemoryGrowthStrategy,
    /// Memory alignment
    pub alignment: MemoryAlignment,
}

/// Memory growth strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryGrowthStrategy {
    /// Fixed size - no growth allowed
    Fixed,
    /// On-demand growth
    OnDemand,
    /// Pre-allocated growth
    PreAllocated,
    /// Streaming growth for large models
    Streaming,
}

/// Memory alignment configuration
#[derive(Debug, Clone)]
pub struct MemoryAlignment {
    /// Data alignment (bytes)
    pub data_alignment: u32,
    /// Function alignment (bytes)
    pub function_alignment: u32,
    /// SIMD alignment (bytes)
    pub simd_alignment: u32,
}

/// WebAssembly export configuration
#[derive(Debug, Clone)]
pub struct WasmExports {
    /// Function exports
    pub functions: Vec<WasmFunctionExport>,
    /// Memory exports
    pub memory: Vec<WasmMemoryExport>,
    /// Global exports
    pub globals: Vec<WasmGlobalExport>,
    /// Table exports
    pub tables: Vec<WasmTableExport>,
}

/// Function export specification
#[derive(Debug, Clone)]
pub struct WasmFunctionExport {
    /// Export name
    pub name: String,
    /// Function signature
    pub signature: WasmSignature,
    /// Documentation
    pub documentation: Option<String>,
    /// Performance hints
    pub performance_hints: Vec<PerformanceHint>,
}

/// WebAssembly function signature
#[derive(Debug, Clone)]
pub struct WasmSignature {
    /// Parameter types
    pub params: Vec<WasmType>,
    /// Return types
    pub returns: Vec<WasmType>,
}

/// WebAssembly primitive types
#[derive(Debug, Clone, PartialEq)]
pub enum WasmType {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// 128-bit SIMD vector (v128)
    V128,
    /// External reference (externref)
    ExternRef,
    /// Function reference (funcref)
    FuncRef,
}

/// Performance hint for exported functions
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceHint {
    /// Function is computationally intensive
    Intensive,
    /// Function should be inlined
    Inline,
    /// Function benefits from SIMD
    SIMDFriendly,
    /// Function is thread-safe
    ThreadSafe,
    /// Function has predictable execution time
    Deterministic,
}

/// Memory export specification
#[derive(Debug, Clone)]
pub struct WasmMemoryExport {
    /// Export name
    pub name: String,
    /// Memory configuration
    pub config: WasmMemoryConfig,
}

/// Global export specification
#[derive(Debug, Clone)]
pub struct WasmGlobalExport {
    /// Export name
    pub name: String,
    /// Global type
    pub global_type: WasmType,
    /// Mutability
    pub mutable: bool,
}

/// Table export specification
#[derive(Debug, Clone)]
pub struct WasmTableExport {
    /// Export name
    pub name: String,
    /// Element type
    pub element_type: WasmType,
    /// Initial size
    pub initial_size: u32,
    /// Maximum size
    pub maximum_size: Option<u32>,
}

/// WebAssembly import configuration
#[derive(Debug, Clone)]
pub struct WasmImports {
    /// Function imports
    pub functions: Vec<WasmFunctionImport>,
    /// Memory imports
    pub memory: Vec<WasmMemoryImport>,
    /// Global imports
    pub globals: Vec<WasmGlobalImport>,
    /// Table imports
    pub tables: Vec<WasmTableImport>,
}

/// Function import specification
#[derive(Debug, Clone)]
pub struct WasmFunctionImport {
    /// Module name
    pub module: String,
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: WasmSignature,
}

/// Memory import specification
#[derive(Debug, Clone)]
pub struct WasmMemoryImport {
    /// Module name
    pub module: String,
    /// Memory name
    pub name: String,
    /// Memory configuration
    pub config: WasmMemoryConfig,
}

/// Global import specification
#[derive(Debug, Clone)]
pub struct WasmGlobalImport {
    /// Module name
    pub module: String,
    /// Global name
    pub name: String,
    /// Global type
    pub global_type: WasmType,
    /// Mutability
    pub mutable: bool,
}

/// Table import specification
#[derive(Debug, Clone)]
pub struct WasmTableImport {
    /// Module name
    pub module: String,
    /// Table name
    pub name: String,
    /// Element type
    pub element_type: WasmType,
    /// Initial size
    pub initial_size: u32,
    /// Maximum size
    pub maximum_size: Option<u32>,
}

/// WebAssembly debugging configuration
#[derive(Debug, Clone)]
pub struct WasmDebugConfig {
    /// Include debug symbols
    pub debug_symbols: bool,
    /// Include source maps
    pub source_maps: bool,
    /// Include function names
    pub function_names: bool,
    /// Enable debug assertions
    pub debug_assertions: bool,
    /// Profiling support
    pub profiling: ProfilingConfig,
}

/// Profiling configuration
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Enable function-level profiling
    pub function_profiling: bool,
    /// Enable memory profiling
    pub memory_profiling: bool,
    /// Enable instruction-level profiling
    pub instruction_profiling: bool,
    /// Profiling output format
    pub output_format: ProfilingFormat,
}

/// Profiling output format
#[derive(Debug, Clone, PartialEq)]
pub enum ProfilingFormat {
    /// Chrome DevTools format
    ChromeDevTools,
    /// Firefox Profiler format
    Firefox,
    /// Custom JSON format
    JSON,
    /// Binary format
    Binary,
}

/// Web integration configuration
#[derive(Debug, Clone)]
pub struct WebIntegrationConfig {
    /// JavaScript/TypeScript bindings
    pub bindings: WebBindingConfig,
    /// Web acceleration
    pub acceleration: WebAccelerationConfig,
    /// Caching strategy
    pub caching: CachingConfig,
    /// Progressive loading
    pub progressive_loading: ProgressiveLoadingConfig,
    /// Web Workers support
    pub workers: WorkerConfig,
}

/// Web binding configuration
#[derive(Debug, Clone)]
pub struct WebBindingConfig {
    /// Target language
    pub target_language: WebBindingLanguage,
    /// Module system
    pub module_system: ModuleSystem,
    /// Type definitions
    pub type_definitions: bool,
    /// Documentation generation
    pub documentation: bool,
    /// Bundle configuration
    pub bundling: BundlingConfig,
}

/// Web binding target language
#[derive(Debug, Clone, PartialEq)]
pub enum WebBindingLanguage {
    /// JavaScript (ES5)
    JavaScript,
    /// Modern JavaScript (ES2020+)
    ModernJavaScript,
    /// TypeScript
    TypeScript,
    /// Both JavaScript and TypeScript
    Both,
}

/// Module system for web bindings
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleSystem {
    /// CommonJS
    CommonJS,
    /// ES Modules
    ESModules,
    /// AMD (RequireJS)
    AMD,
    /// UMD (Universal Module Definition)
    UMD,
    /// IIFE (Immediately Invoked Function Expression)
    IIFE,
}

/// Bundling configuration
#[derive(Debug, Clone)]
pub struct BundlingConfig {
    /// Enable bundling
    pub enable: bool,
    /// Bundle format
    pub format: BundleFormat,
    /// Minification
    pub minify: bool,
    /// Tree shaking
    pub tree_shaking: bool,
    /// Code splitting
    pub code_splitting: bool,
}

/// Bundle format
#[derive(Debug, Clone, PartialEq)]
pub enum BundleFormat {
    /// Single file bundle
    Single,
    /// Multiple chunks
    Chunked,
    /// Streaming bundle
    Streaming,
}

/// Web acceleration configuration
#[derive(Debug, Clone)]
pub struct WebAccelerationConfig {
    /// WebGL support
    pub webgl: WebGLConfig,
    /// WebGPU support
    pub webgpu: WebGPUConfig,
    /// SIMD.js fallback
    pub simd_js: bool,
    /// Parallel execution
    pub parallel: ParallelConfig,
}

/// WebGL configuration
#[derive(Debug, Clone)]
pub struct WebGLConfig {
    /// Enable WebGL acceleration
    pub enable: bool,
    /// WebGL version (1 or 2)
    pub version: u8,
    /// Shader optimization
    pub shader_optimization: bool,
    /// Texture formats
    pub texture_formats: Vec<TextureFormat>,
}

/// Texture format for WebGL
#[derive(Debug, Clone, PartialEq)]
pub enum TextureFormat {
    /// RGBA8
    RGBA8,
    /// RGB8
    RGB8,
    /// RG8
    RG8,
    /// R8
    R8,
    /// RGBA32F
    RGBA32F,
    /// RGB32F
    RGB32F,
    /// RG32F
    RG32F,
    /// R32F
    R32F,
}

/// WebGPU configuration
#[derive(Debug, Clone)]
pub struct WebGPUConfig {
    /// Enable WebGPU acceleration
    pub enable: bool,
    /// Compute shader support
    pub compute_shaders: bool,
    /// Memory optimization
    pub memory_optimization: bool,
    /// Pipeline caching
    pub pipeline_caching: bool,
}

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Enable Web Workers
    pub web_workers: bool,
    /// Maximum worker threads
    pub max_workers: Option<usize>,
    /// Shared memory usage
    pub shared_memory: bool,
    /// Work stealing
    pub work_stealing: bool,
}

/// Caching configuration
#[derive(Debug, Clone)]
pub struct CachingConfig {
    /// Enable caching
    pub enable: bool,
    /// Cache strategy
    pub strategy: CacheStrategy,
    /// Cache storage
    pub storage: CacheStorage,
    /// Cache TTL (time-to-live) in seconds
    pub ttl_seconds: Option<u64>,
    /// Version management
    pub versioning: VersioningStrategy,
}

/// Cache strategy
#[derive(Debug, Clone, PartialEq)]
pub enum CacheStrategy {
    /// Cache everything
    Aggressive,
    /// Cache only frequently used items
    Conservative,
    /// Least Recently Used (LRU)
    LRU,
    /// First In, First Out (FIFO)
    FIFO,
    /// Custom strategy
    Custom(String),
}

/// Cache storage backend
#[derive(Debug, Clone, PartialEq)]
pub enum CacheStorage {
    /// Browser memory
    Memory,
    /// Local Storage
    LocalStorage,
    /// Session Storage
    SessionStorage,
    /// IndexedDB
    IndexedDB,
    /// Cache API
    CacheAPI,
    /// Custom storage
    Custom(String),
}

/// Versioning strategy for cache
#[derive(Debug, Clone, PartialEq)]
pub enum VersioningStrategy {
    /// Use semantic versioning
    Semantic,
    /// Use hash-based versioning
    Hash,
    /// Use timestamp versioning
    Timestamp,
    /// No versioning
    None,
}

/// Progressive loading configuration
#[derive(Debug, Clone)]
pub struct ProgressiveLoadingConfig {
    /// Enable progressive loading
    pub enable: bool,
    /// Loading strategy
    pub strategy: LoadingStrategy,
    /// Chunk size in bytes
    pub chunk_size: usize,
    /// Preloading configuration
    pub preloading: PreloadingConfig,
    /// Streaming support
    pub streaming: bool,
}

/// Loading strategy
#[derive(Debug, Clone, PartialEq)]
pub enum LoadingStrategy {
    /// Load everything at once
    Eager,
    /// Load on demand
    Lazy,
    /// Load based on priority
    Priority,
    /// Load based on viewport
    Viewport,
    /// Custom loading logic
    Custom(String),
}

/// Preloading configuration
#[derive(Debug, Clone)]
pub struct PreloadingConfig {
    /// Enable preloading
    pub enable: bool,
    /// Preload percentage (0.0 to 1.0)
    pub percentage: f64,
    /// Preload on idle
    pub on_idle: bool,
    /// Preload based on user interaction
    pub on_interaction: bool,
}

/// Web Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Enable Web Workers
    pub enable: bool,
    /// Worker type
    pub worker_type: WorkerType,
    /// Message passing strategy
    pub messaging: MessagingStrategy,
    /// Worker pool configuration
    pub pool: WorkerPoolConfig,
}

/// Web Worker type
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerType {
    /// Dedicated Worker
    Dedicated,
    /// Shared Worker
    Shared,
    /// Service Worker
    Service,
}

/// Message passing strategy for workers
#[derive(Debug, Clone, PartialEq)]
pub enum MessagingStrategy {
    /// Copy data (default)
    Copy,
    /// Transfer ownership
    Transfer,
    /// Shared memory
    Shared,
}

/// Worker pool configuration
#[derive(Debug, Clone)]
pub struct WorkerPoolConfig {
    /// Minimum workers
    pub min_workers: usize,
    /// Maximum workers
    pub max_workers: usize,
    /// Idle timeout in milliseconds
    pub idle_timeout_ms: u64,
    /// Task queue size
    pub queue_size: usize,
}

/// WebAssembly compiler and deployment manager
pub struct WasmCompiler<F: Float + Debug + ndarray::ScalarOperand> {
    /// Model to compile
    #[allow(dead_code)]
    model: Sequential<F>,
    /// Compilation configuration
    #[allow(dead_code)]
    config: WasmCompilationConfig,
    /// Web integration configuration
    web_config: WebIntegrationConfig,
    /// Package metadata
    #[allow(dead_code)]
    metadata: PackageMetadata,
    /// Output directory
    output_dir: PathBuf,
}

/// Compilation result
#[derive(Debug, Clone)]
pub struct WasmCompilationResult {
    /// Generated WASM module path
    pub wasm_module: PathBuf,
    /// JavaScript/TypeScript bindings
    pub bindings: Vec<PathBuf>,
    /// Web integration files
    pub web_files: Vec<PathBuf>,
    /// Documentation files
    pub documentation: Vec<PathBuf>,
    /// Bundle information
    pub bundle_info: BundleInfo,
}

/// Bundle information
#[derive(Debug, Clone)]
pub struct BundleInfo {
    /// Total bundle size in bytes
    pub total_size: usize,
    /// WASM module size in bytes
    pub wasm_size: usize,
    /// JavaScript size in bytes
    pub js_size: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Load time estimation in milliseconds
    pub estimated_load_time_ms: u64,
}

impl<
        F: Float + Debug + 'static + num_traits::FromPrimitive + ndarray::ScalarOperand + Send + Sync,
    > WasmCompiler<F>
{
    /// Create a new WASM compiler
    pub fn new(
        model: Sequential<F>,
        config: WasmCompilationConfig,
        web_config: WebIntegrationConfig,
        metadata: PackageMetadata,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            model,
            config,
            web_config,
            metadata,
            output_dir,
        }
    }

    /// Compile model to WebAssembly
    pub fn compile(&self) -> Result<WasmCompilationResult> {
        // Create output directory structure
        self.create_directory_structure()?;

        // Generate WASM module
        let wasm_module = self.generate_wasm_module()?;

        // Generate bindings
        let bindings = self.generate_bindings()?;

        // Generate web integration files
        let web_files = self.generate_web_files()?;

        // Generate documentation
        let documentation = self.generate_documentation()?;

        // Calculate bundle information
        let bundle_info = self.calculate_bundle_info(&wasm_module, &bindings)?;

        Ok(WasmCompilationResult {
            wasm_module,
            bindings,
            web_files,
            documentation,
            bundle_info,
        })
    }

    fn create_directory_structure(&self) -> Result<()> {
        let dirs = vec![
            "wasm", "js", "ts", "docs", "examples", "tests", "build", "dist",
        ];

        for dir in dirs {
            let path = self.output_dir.join(dir);
            fs::create_dir_all(&path).map_err(|e| {
                NeuralError::IOError(format!(
                    "Failed to create directory {}: {}",
                    path.display(),
                    e
                ))
            })?;
        }

        Ok(())
    }

    fn generate_wasm_module(&self) -> Result<PathBuf> {
        let wasm_path = self.output_dir.join("wasm").join("model.wasm");

        // Stub implementation - in real code, this would use wasm-pack or similar tools
        let wasm_header = vec![
            0x00, 0x61, 0x73, 0x6d, // Magic number (\0asm)
            0x01, 0x00, 0x00, 0x00, // Version
        ];

        fs::write(&wasm_path, wasm_header).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(wasm_path)
    }

    fn generate_bindings(&self) -> Result<Vec<PathBuf>> {
        let mut bindings = Vec::new();

        match self.web_config.bindings.target_language {
            WebBindingLanguage::JavaScript => {
                let js_path = self.generate_javascript_bindings()?;
                bindings.push(js_path);
            }
            WebBindingLanguage::TypeScript => {
                let ts_path = self.generate_typescript_bindings()?;
                bindings.push(ts_path);
            }
            WebBindingLanguage::ModernJavaScript => {
                let js_path = self.generate_modern_javascript_bindings()?;
                bindings.push(js_path);
            }
            WebBindingLanguage::Both => {
                let js_path = self.generate_javascript_bindings()?;
                bindings.push(js_path);
                let ts_path = self.generate_typescript_bindings()?;
                bindings.push(ts_path);
            }
        }

        Ok(bindings)
    }

    fn generate_javascript_bindings(&self) -> Result<PathBuf> {
        let js_path = self.output_dir.join("js").join("scirs2-model.js");

        let js_content = r#"/**
 * SciRS2 Model WebAssembly Bindings
 * 
 * This module provides JavaScript bindings for running SciRS2 neural network models
 * in the browser using WebAssembly.
 */

class SciRS2Model {
    constructor() {
        this.module = null;
        this.memory = null;
        this.exports = null;
        this.isInitialized = false;
    }

    /**
     * Initialize the WASM module
     * @param {string|ArrayBuffer} wasmSource - Path to WASM file or ArrayBuffer
     * @param {Object} options - Initialization options
     * @returns {Promise<void>}
     */
    async initialize(wasmSource, options = {}) {
        try {
            let wasmBytes;
            
            if (typeof wasmSource === 'string') {
                const response = await fetch(wasmSource);
                wasmBytes = await response.arrayBuffer();
            } else {
                wasmBytes = wasmSource;
            }

            const wasmModule = await WebAssembly.instantiate(wasmBytes, {
                env: {
                    memory: new WebAssembly.Memory({ 
                        initial: options.initialMemory || 256,
                        maximum: options.maximumMemory || 1024 
                    }),
                    abort: () => {
                        throw new Error('WASM execution aborted');
                    }
                }
            });

            this.module = wasmModule.module;
            this.exports = wasmModule.instance.exports;
            this.memory = this.exports.memory || wasmModule.instance.exports.memory;
            this.isInitialized = true;

            console.log('SciRS2 Model initialized successfully');
        } catch (error) {
            throw new Error(`Failed to initialize WASM module: ${error.message}`);
        }
    }

    /**
     * Run inference on input data
     * @param {Float32Array|Array} inputData - Input tensor data
     * @param {Array} inputShape - Shape of input tensor
     * @returns {Promise<Float32Array>} Output tensor data
     */
    async predict(inputData, inputShape) {
        if (!this.isInitialized) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        try {
            // Convert input to Float32Array if needed
            const input = inputData instanceof Float32Array ? inputData : new Float32Array(inputData);
            
            // Allocate memory for input
            const inputSize = input.length * 4; // 4 bytes per float32
            const inputPtr = this.exports.allocate(inputSize);
            
            if (!inputPtr) {
                throw new Error('Failed to allocate memory for input');
            }

            // Copy input data to WASM memory
            const inputView = new Float32Array(this.memory.buffer, inputPtr, input.length);
            inputView.set(input);

            // Allocate memory for output (assuming same size for now)
            const outputSize = inputSize; // This would be calculated based on model architecture
            const outputPtr = this.exports.allocate(outputSize);
            
            if (!outputPtr) {
                this.exports.deallocate(inputPtr);
                throw new Error('Failed to allocate memory for output');
            }

            // Run inference
            const result = this.exports.predict(inputPtr, outputPtr, input.length);
            
            if (result !== 0) {
                this.exports.deallocate(inputPtr);
                this.exports.deallocate(outputPtr);
                throw new Error(`Inference failed with error code: ${result}`);
            }

            // Copy output data from WASM memory
            const outputView = new Float32Array(this.memory.buffer, outputPtr, input.length);
            const output = new Float32Array(outputView);

            // Clean up memory
            this.exports.deallocate(inputPtr);
            this.exports.deallocate(outputPtr);

            return output;
        } catch (error) {
            throw new Error(`Prediction failed: ${error.message}`);
        }
    }

    /**
     * Get model information
     * @returns {Object} Model metadata
     */
    getModelInfo() {
        if (!this.isInitialized) {
            throw new Error('Model not initialized');
        }

        return {
            version: this.exports.get_version ? this.exports.get_version() : 'unknown',
            inputShape: [1, 10], // Stub - would be extracted from model
            outputShape: [1, 1], // Stub - would be extracted from model
            parameters: this.exports.get_parameter_count ? this.exports.get_parameter_count() : 0
        };
    }

    /**
     * Check if model supports batch inference
     * @returns {boolean}
     */
    supportsBatchInference() {
        return this.exports.predict_batch !== undefined;
    }

    /**
     * Run batch inference
     * @param {Array<Float32Array>} inputs - Array of input tensors
     * @param {Array} inputShape - Shape of each input tensor
     * @returns {Promise<Array<Float32Array>>} Array of output tensors
     */
    async predictBatch(inputs, inputShape) {
        if (!this.supportsBatchInference()) {
            // Fallback to sequential prediction
            const outputs = [];
            for (const input of inputs) {
                const output = await this.predict(input, inputShape);
                outputs.push(output);
            }
            return outputs;
        }

        // Batch inference implementation would go here
        throw new Error('Batch inference not yet implemented');
    }

    /**
     * Dispose of resources
     */
    dispose() {
        if (this.isInitialized) {
            // Clean up any allocated resources
            if (this.exports.cleanup) {
                this.exports.cleanup();
            }
            this.module = null;
            this.exports = null;
            this.memory = null;
            this.isInitialized = false;
        }
    }
}

// Utility functions

/**
 * Load model with caching support
 * @param {string} modelUrl - URL to model WASM file
 * @param {Object} options - Loading options
 * @returns {Promise<SciRS2Model>}
 */
async function loadModel(modelUrl, options = {}) {
    const model = new SciRS2Model();
    
    let wasmSource;
    
    if (options.useCache && 'caches' in window) {
        try {
            const cache = await caches.open('scirs2-models');
            const cachedResponse = await cache.match(modelUrl);
            
            if (cachedResponse) {
                wasmSource = await cachedResponse.arrayBuffer();
                console.log('Loaded model from cache');
            } else {
                const response = await fetch(modelUrl);
                await cache.put(modelUrl, response.clone());
                wasmSource = await response.arrayBuffer();
                console.log('Loaded model from network and cached');
            }
        } catch (error) {
            console.warn('Cache operation failed, falling back to direct fetch:', error);
            const response = await fetch(modelUrl);
            wasmSource = await response.arrayBuffer();
        }
    } else {
        const response = await fetch(modelUrl);
        wasmSource = await response.arrayBuffer();
    }
    
    await model.initialize(wasmSource, options);
    return model;
}

/**
 * Check WebAssembly support
 * @returns {Object} Support information
 */
function checkWasmSupport() {
    const support = {
        basic: typeof WebAssembly === 'object',
        simd: false,
        threads: false,
        bulkMemory: false
    };

    if (support.basic) {
        // Check for SIMD support
        try {
            support.simd = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0f, 0xfd, 0x0f, 0x0b
            ]));
        } catch (e) {
            support.simd = false;
        }

        // Check for threading support
        support.threads = typeof SharedArrayBuffer === 'function';

        // Check for bulk memory support
        try {
            support.bulkMemory = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x00, 0x0b
            ]));
        } catch (e) {
            support.bulkMemory = false;
        }
    }

    return support;
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = { SciRS2Model, loadModel, checkWasmSupport };
} else if (typeof define === 'function' && define.amd) {
    // AMD
    define([], function() {
        return { SciRS2Model, loadModel, checkWasmSupport };
    });
} else {
    // Global
    window.SciRS2 = { SciRS2Model, loadModel, checkWasmSupport };
}
"#;

        fs::write(&js_path, js_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(js_path)
    }

    fn generate_typescript_bindings(&self) -> Result<PathBuf> {
        let ts_path = self.output_dir.join("ts").join("scirs2-model.ts");

        let ts_content = r#"/**
 * SciRS2 Model WebAssembly TypeScript Bindings
 * 
 * This module provides TypeScript bindings for running SciRS2 neural network models
 * in the browser using WebAssembly with full type safety.
 */

interface WasmSupport {
    basic: boolean;
    simd: boolean;
    threads: boolean;
    bulkMemory: boolean;
}

interface ModelInfo {
    version: string;
    inputShape: number[];
    outputShape: number[];
    parameters: number;
}

interface InitializationOptions {
    initialMemory?: number;
    maximumMemory?: number;
    enableSIMD?: boolean;
    enableThreads?: boolean;
}

interface LoadingOptions extends InitializationOptions {
    useCache?: boolean;
    progressCallback?: (progress: number) => void;
    timeout?: number;
}

interface PredictionOptions {
    batchSize?: number;
    timeout?: number;
    useWorker?: boolean;
}

// Type definitions for WASM exports
interface WasmExports {
    memory: WebAssembly.Memory;
    allocate: (size: number) => number;
    deallocate: (ptr: number) => void;
    predict: (inputPtr: number, outputPtr: number, size: number) => number;
    predict_batch?: (inputPtr: number, outputPtr: number, batchSize: number, size: number) => number;
    get_version?: () => number;
    get_parameter_count?: () => number;
    cleanup?: () => void;
}

class SciRS2Model {
    private module: WebAssembly.Module | null = null;
    private exports: WasmExports | null = null;
    private memory: WebAssembly.Memory | null = null;
    private isInitialized = false;

    /**
     * Initialize the WASM module
     */
    async initialize(wasmSource: string | ArrayBuffer, options: InitializationOptions = {}): Promise<void> {
        try {
            let wasmBytes: ArrayBuffer;
            
            if (typeof wasmSource === 'string') {
                const response = await fetch(wasmSource);
                if (!response.ok) {
                    throw new Error(`Failed to fetch WASM module: ${response.statusText}`);
                }
                wasmBytes = await response.arrayBuffer();
            } else {
                wasmBytes = wasmSource;
            }

            const imports = {
                env: {
                    memory: new WebAssembly.Memory({ 
                        initial: options.initialMemory || 256,
                        maximum: options.maximumMemory || 1024,
                        shared: options.enableThreads || false
                    }),
                    abort: (): never => {
                        throw new Error('WASM execution aborted');
                    }
                }
            };

            const wasmModule = await WebAssembly.instantiate(wasmBytes, imports);

            this.module = wasmModule.module;
            this.exports = wasmModule.instance.exports as WasmExports;
            this.memory = this.exports.memory || imports.env.memory;
            this.isInitialized = true;

            console.log('SciRS2 Model initialized successfully');
        } catch (error) {
            throw new Error(`Failed to initialize WASM module: ${(error as Error).message}`);
        }
    }

    /**
     * Run inference on input data
     */
    async predict(
        inputData: Float32Array | number[], 
        inputShape: number[], 
        options: PredictionOptions = {}
    ): Promise<Float32Array> {
        if (!this.isInitialized || !this.exports || !this.memory) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        try {
            // Convert input to Float32Array if needed
            const input = inputData instanceof Float32Array ? inputData : new Float32Array(inputData);
            
            // Validate input shape
            const expectedSize = inputShape.reduce((a, b) => a * b, 1);
            if (input.length !== expectedSize) {
                throw new Error(`Input size ${input.length} doesn't match expected size ${expectedSize}`);
            }

            // Allocate memory for input
            const inputSize = input.length * 4; // 4 bytes per float32
            const inputPtr = this.exports.allocate(inputSize);
            
            if (!inputPtr) {
                throw new Error('Failed to allocate memory for input');
            }

            try {
                // Copy input data to WASM memory
                const inputView = new Float32Array(this.memory.buffer, inputPtr, input.length);
                inputView.set(input);

                // Calculate output size (this would be determined by model architecture)
                const outputSize = this.calculateOutputSize(inputShape);
                const outputPtr = this.exports.allocate(outputSize * 4);
                
                if (!outputPtr) {
                    throw new Error('Failed to allocate memory for output');
                }

                try {
                    // Run inference with timeout
                    const result = await this.executeWithTimeout(
                        () => this.exports!.predict(inputPtr, outputPtr, input.length),
                        options.timeout || 5000
                    );
                    
                    if (result !== 0) {
                        throw new Error(`Inference failed with error code: ${result}`);
                    }

                    // Copy output data from WASM memory
                    const outputView = new Float32Array(this.memory.buffer, outputPtr, outputSize);
                    return new Float32Array(outputView);
                } finally {
                    this.exports.deallocate(outputPtr);
                }
            } finally {
                this.exports.deallocate(inputPtr);
            }
        } catch (error) {
            throw new Error(`Prediction failed: ${(error as Error).message}`);
        }
    }

    /**
     * Run batch inference
     */
    async predictBatch(
        inputs: (Float32Array | number[])[], 
        inputShape: number[], 
        options: PredictionOptions = {}
    ): Promise<Float32Array[]> {
        if (!this.supportsBatchInference()) {
            // Fallback to sequential prediction
            const outputs: Float32Array[] = [];
            for (const input of inputs) {
                const output = await this.predict(input, inputShape, options);
                outputs.push(output);
            }
            return outputs;
        }

        // Batch inference implementation would go here
        throw new Error('Batch inference not yet implemented');
    }

    /**
     * Get model information
     */
    getModelInfo(): ModelInfo {
        if (!this.isInitialized || !this.exports) {
            throw new Error('Model not initialized');
        }

        return {
            version: this.exports.get_version ? this.exports.get_version().toString() : 'unknown',
            inputShape: [1, 10], // Stub - would be extracted from model
            outputShape: [1, 1], // Stub - would be extracted from model
            parameters: this.exports.get_parameter_count ? this.exports.get_parameter_count() : 0
        };
    }

    /**
     * Check if model supports batch inference
     */
    supportsBatchInference(): boolean {
        return this.exports?.predict_batch !== undefined;
    }

    /**
     * Check if model is initialized
     */
    isReady(): boolean {
        return this.isInitialized;
    }

    /**
     * Dispose of resources
     */
    dispose(): void {
        if (this.isInitialized && this.exports) {
            // Clean up any allocated resources
            if (this.exports.cleanup) {
                this.exports.cleanup();
            }
            this.module = null;
            this.exports = null;
            this.memory = null;
            this.isInitialized = false;
        }
    }

    private calculateOutputSize(inputShape: number[]): number {
        // Stub implementation - would be based on model architecture
        return inputShape.reduce((a, b) => a * b, 1);
    }

    private async executeWithTimeout<T>(fn: () => T, timeoutMs: number): Promise<T> {
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                reject(new Error(`Operation timed out after ${timeoutMs}ms`));
            }, timeoutMs);

            try {
                const result = fn();
                clearTimeout(timer);
                resolve(result);
            } catch (error) {
                clearTimeout(timer);
                reject(error);
            }
        });
    }
}

// Utility functions

/**
 * Load model with caching support
 */
export async function loadModel(modelUrl: string, options: LoadingOptions = {}): Promise<SciRS2Model> {
    const model = new SciRS2Model();
    
    let wasmSource: ArrayBuffer;
    
    if (options.useCache && 'caches' in window) {
        try {
            const cache = await caches.open('scirs2-models');
            const cachedResponse = await cache.match(modelUrl);
            
            if (cachedResponse) {
                wasmSource = await cachedResponse.arrayBuffer();
                console.log('Loaded model from cache');
            } else {
                const response = await fetch(modelUrl);
                if (!response.ok) {
                    throw new Error(`Failed to fetch model: ${response.statusText}`);
                }
                await cache.put(modelUrl, response.clone());
                wasmSource = await response.arrayBuffer();
                console.log('Loaded model from network and cached');
            }
        } catch (error) {
            console.warn('Cache operation failed, falling back to direct fetch:', error);
            const response = await fetch(modelUrl);
            wasmSource = await response.arrayBuffer();
        }
    } else {
        const response = await fetch(modelUrl);
        wasmSource = await response.arrayBuffer();
    }
    
    await model.initialize(wasmSource, options);
    return model;
}

/**
 * Check WebAssembly support
 */
export function checkWasmSupport(): WasmSupport {
    const support: WasmSupport = {
        basic: typeof WebAssembly === 'object',
        simd: false,
        threads: false,
        bulkMemory: false
    };

    if (support.basic) {
        // Check for SIMD support
        try {
            support.simd = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0f, 0xfd, 0x0f, 0x0b
            ]));
        } catch (e) {
            support.simd = false;
        }

        // Check for threading support
        support.threads = typeof SharedArrayBuffer === 'function';

        // Check for bulk memory support
        try {
            support.bulkMemory = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x00, 0x0b
            ]));
        } catch (e) {
            support.bulkMemory = false;
        }
    }

    return support;
}

// Re-export the main class
export { SciRS2Model };
export type { ModelInfo, WasmSupport, InitializationOptions, LoadingOptions, PredictionOptions };
"#;

        fs::write(&ts_path, ts_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(ts_path)
    }

    fn generate_modern_javascript_bindings(&self) -> Result<PathBuf> {
        let js_path = self.output_dir.join("js").join("scirs2-model.mjs");

        let js_content = r#"/**
 * SciRS2 Model WebAssembly Modern JavaScript Bindings (ES2020+)
 * 
 * This module provides modern JavaScript bindings with ES modules, async/await,
 * and other ES2020+ features for running SciRS2 neural network models.
 */

class SciRS2Model {
    #module = null;
    #exports = null;
    #memory = null;
    #isInitialized = false;

    /**
     * Initialize the WASM module
     * @param {string|ArrayBuffer} wasmSource - Path to WASM file or ArrayBuffer
     * @param {Object} options - Initialization options
     */
    async initialize(wasmSource, options = {}) {
        try {
            let wasmBytes;
            
            if (typeof wasmSource === 'string') {
                const response = await fetch(wasmSource);
                if (!response.ok) {
                    throw new Error(`Failed to fetch WASM module: ${response.statusText}`);
                }
                wasmBytes = await response.arrayBuffer();
            } else {
                wasmBytes = wasmSource;
            }

            const imports = {
                env: {
                    memory: new WebAssembly.Memory({ 
                        initial: options.initialMemory ?? 256,
                        maximum: options.maximumMemory ?? 1024,
                        shared: options.enableThreads ?? false
                    }),
                    abort: () => {
                        throw new Error('WASM execution aborted');
                    }
                }
            };

            const wasmModule = await WebAssembly.instantiate(wasmBytes, imports);

            this.#module = wasmModule.module;
            this.#exports = wasmModule.instance.exports;
            this.#memory = this.#exports.memory ?? imports.env.memory;
            this.#isInitialized = true;

            console.log('SciRS2 Model initialized successfully');
        } catch (error) {
            throw new Error(`Failed to initialize WASM module: ${error.message}`);
        }
    }

    /**
     * Run inference on input data using modern async patterns
     * @param {Float32Array|Array} inputData - Input tensor data
     * @param {Array} inputShape - Shape of input tensor
     * @param {Object} options - Prediction options
     */
    async predict(inputData, inputShape, options = {}) {
        if (!this.#isInitialized) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        const input = inputData instanceof Float32Array ? inputData : new Float32Array(inputData);
        
        // Use structured concurrency pattern
        const prediction = await this.#withMemoryManagement(async (allocate, deallocate) => {
            // Allocate input memory
            const inputSize = input.length * 4;
            const inputPtr = allocate(inputSize);
            
            // Copy input data
            const inputView = new Float32Array(this.#memory.buffer, inputPtr, input.length);
            inputView.set(input);

            // Allocate output memory
            const outputSize = this.#calculateOutputSize(inputShape);
            const outputPtr = allocate(outputSize * 4);
            
            // Run inference with timeout
            const result = await this.#executeWithTimeout(
                () => this.#exports.predict(inputPtr, outputPtr, input.length),
                options.timeout ?? 5000
            );
            
            if (result !== 0) {
                throw new Error(`Inference failed with error code: ${result}`);
            }

            // Copy and return output
            const outputView = new Float32Array(this.#memory.buffer, outputPtr, outputSize);
            return new Float32Array(outputView);
        });

        return prediction;
    }

    /**
     * Run batch inference with parallel execution
     * @param {Array<Float32Array>} inputs - Array of input tensors
     * @param {Array} inputShape - Shape of each input tensor
     * @param {Object} options - Prediction options
     */
    async predictBatch(inputs, inputShape, options = {}) {
        const maxConcurrency = options.maxConcurrency ?? 4;
        
        if (!this.supportsBatchInference()) {
            // Use Promise.allSettled for parallel execution with concurrency limit
            const results = [];
            for (let i = 0; i < inputs.length; i += maxConcurrency) {
                const batch = inputs.slice(i, i + maxConcurrency);
                const batchPromises = batch.map(input => this.predict(input, inputShape, options));
                const batchResults = await Promise.allSettled(batchPromises);
                
                results.push(...batchResults.map(result => {
                    if (result.status === 'fulfilled') {
                        return result.value;
                    } else {
                        throw result.reason;
                    }
                }));
            }
            return results;
        }

        // Native batch inference implementation would go here
        throw new Error('Native batch inference not yet implemented');
    }

    /**
     * Get model information with async metadata loading
     */
    async getModelInfo() {
        if (!this.#isInitialized) {
            throw new Error('Model not initialized');
        }

        // This could be enhanced to load metadata asynchronously
        return {
            version: this.#exports.get_version?.() ?? 'unknown',
            inputShape: [1, 10], // Would be extracted from model
            outputShape: [1, 1], // Would be extracted from model
            parameters: this.#exports.get_parameter_count?.() ?? 0,
            memoryUsage: this.#memory.buffer.byteLength
        };
    }

    /**
     * Check capabilities
     */
    get capabilities() {
        return {
            batchInference: this.supportsBatchInference(),
            streaming: false, // Would be determined by model
            webgl: false, // Would be determined by compilation options
            simd: this.#checkSIMDSupport()
        };
    }

    get isReady() {
        return this.#isInitialized;
    }

    supportsBatchInference() {
        return this.#exports?.predict_batch !== undefined;
    }

    /**
     * Dispose with proper cleanup
     */
    dispose() {
        if (this.#isInitialized) {
            this.#exports?.cleanup?.();
            this.#module = null;
            this.#exports = null;
            this.#memory = null;
            this.#isInitialized = false;
        }
    }

    // Private methods using private fields syntax

    #calculateOutputSize(inputShape) {
        return inputShape.reduce((a, b) => a * b, 1);
    }

    async #executeWithTimeout(fn, timeoutMs) {
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs);
        });

        const operationPromise = new Promise((resolve, reject) => {
            try {
                const result = fn();
                resolve(result);
            } catch (error) {
                reject(error);
            }
        });

        return Promise.race([operationPromise, timeoutPromise]);
    }

    async #withMemoryManagement(fn) {
        const allocatedPtrs = [];
        
        const allocate = (size) => {
            const ptr = this.#exports.allocate(size);
            if (ptr) allocatedPtrs.push(ptr);
            return ptr;
        };

        const deallocate = (ptr) => {
            const index = allocatedPtrs.indexOf(ptr);
            if (index !== -1) {
                allocatedPtrs.splice(index, 1);
                this.#exports.deallocate(ptr);
            }
        };

        try {
            return await fn(allocate, deallocate);
        } finally {
            // Ensure all memory is cleaned up
            allocatedPtrs.forEach(ptr => this.#exports.deallocate(ptr));
        }
    }

    #checkSIMDSupport() {
        try {
            return WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
                0x03, 0x02, 0x01, 0x00,
                0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0f, 0xfd, 0x0f, 0x0b
            ]));
        } catch {
            return false;
        }
    }
}

// Utility functions with modern patterns

/**
 * Load model with enhanced caching and progress
 */
export async function loadModel(modelUrl, options = {}) {
    const model = new SciRS2Model();
    
    const wasmSource = await loadWithProgress(modelUrl, {
        useCache: options.useCache ?? true,
        onProgress: options.onProgress
    });
    
    await model.initialize(wasmSource, options);
    return model;
}

/**
 * Enhanced loading with progress tracking
 */
async function loadWithProgress(url, { useCache = true, onProgress } = {}) {
    if (useCache && 'caches' in window) {
        const cache = await caches.open('scirs2-models-v1');
        const cachedResponse = await cache.match(url);
        
        if (cachedResponse) {
            onProgress?.(100);
            return await cachedResponse.arrayBuffer();
        }
    }

    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    
    if (total === 0 || !onProgress) {
        const arrayBuffer = await response.arrayBuffer();
        onProgress?.(100);
        return arrayBuffer;
    }

    // Stream with progress tracking
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        chunks.push(value);
        loaded += value.length;
        onProgress((loaded / total) * 100);
    }

    // Combine chunks
    const arrayBuffer = new ArrayBuffer(loaded);
    const uint8Array = new Uint8Array(arrayBuffer);
    let offset = 0;
    
    for (const chunk of chunks) {
        uint8Array.set(chunk, offset);
        offset += chunk.length;
    }

    // Cache the result
    if (useCache && 'caches' in window) {
        try {
            const cache = await caches.open('scirs2-models-v1');
            const cacheResponse = new Response(arrayBuffer.slice(0));
            await cache.put(url, cacheResponse);
        } catch (error) {
            console.warn('Failed to cache model:', error);
        }
    }

    return arrayBuffer;
}

/**
 * Check WebAssembly support with detailed capabilities
 */
export function checkWasmSupport() {
    const support = {
        basic: typeof WebAssembly === 'object',
        simd: false,
        threads: false,
        bulkMemory: false,
        multiValue: false,
        referenceTypes: false
    };

    if (!support.basic) return support;

    const testCases = [
        {
            name: 'simd',
            bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                   0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
                   0x03, 0x02, 0x01, 0x00,
                   0x0a, 0x0a, 0x01, 0x08, 0x00, 0xfd, 0x0f, 0xfd, 0x0f, 0x0b]
        },
        {
            name: 'bulkMemory',
            bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                   0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                   0x03, 0x02, 0x01, 0x00,
                   0x0a, 0x09, 0x01, 0x07, 0x00, 0x41, 0x00, 0xfc, 0x0a, 0x00, 0x0b]
        }
    ];

    for (const { name, bytes } of testCases) {
        try {
            support[name] = WebAssembly.validate(new Uint8Array(bytes));
        } catch {
            support[name] = false;
        }
    }

    support.threads = typeof SharedArrayBuffer === 'function';

    return support;
}

export { SciRS2Model };
"#;

        fs::write(&js_path, js_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(js_path)
    }

    fn generate_web_files(&self) -> Result<Vec<PathBuf>> {
        let mut web_files = Vec::new();

        // Generate package.json
        let package_json_path = self.generate_package_json()?;
        web_files.push(package_json_path);

        // Generate webpack config
        let webpack_path = self.generate_webpack_config()?;
        web_files.push(webpack_path);

        // Generate example HTML
        let html_path = self.generate_example_html()?;
        web_files.push(html_path);

        // Generate web worker
        let worker_path = self.generate_web_worker()?;
        web_files.push(worker_path);

        Ok(web_files)
    }

    fn generate_package_json(&self) -> Result<PathBuf> {
        let package_json_path = self.output_dir.join("package.json");

        let package_json = serde_json::json!({
            "name": "@scirs2/neural-wasm",
            "version": "1.0.0",
            "description": "SciRS2 Neural Network WebAssembly bindings",
            "main": "js/scirs2-model.js",
            "module": "js/scirs2-model.mjs",
            "types": "ts/scirs2-model.d.ts",
            "files": [
                "wasm/",
                "js/",
                "ts/",
                "README.md"
            ],
            "scripts": {
                "build": "webpack --mode=production",
                "dev": "webpack serve --mode=development",
                "test": "jest",
                "typecheck": "tsc --noEmit",
                "lint": "eslint js/ ts/"
            },
            "keywords": [
                "neural-network",
                "machine-learning",
                "webassembly",
                "wasm",
                "scirs2"
            ],
            "author": "SciRS2 Team",
            "license": "MIT",
            "devDependencies": {
                "@types/jest": "^29.0.0",
                "@typescript-eslint/eslint-plugin": "^5.0.0",
                "@typescript-eslint/parser": "^5.0.0",
                "eslint": "^8.0.0",
                "jest": "^29.0.0",
                "typescript": "^4.9.0",
                "webpack": "^5.0.0",
                "webpack-cli": "^4.0.0",
                "webpack-dev-server": "^4.0.0"
            },
            "browser": {
                "./js/scirs2-model.js": "./js/scirs2-model.js"
            },
            "exports": {
                ".": {
                    "import": "./js/scirs2-model.mjs",
                    "require": "./js/scirs2-model.js",
                    "types": "./ts/scirs2-model.d.ts"
                },
                "./wasm": "./wasm/model.wasm"
            }
        });

        let package_json_str = serde_json::to_string_pretty(&package_json)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;

        fs::write(&package_json_path, package_json_str)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(package_json_path)
    }

    fn generate_webpack_config(&self) -> Result<PathBuf> {
        let webpack_path = self.output_dir.join("webpack.config.js");

        let webpack_content = r#"const path = require('path');

module.exports = {
    entry: './js/scirs2-model.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'scirs2-model.bundle.js',
        library: 'SciRS2',
        libraryTarget: 'umd',
        globalObject: 'this'
    },
    mode: 'production',
    module: {
        rules: [
            {
                test: /\.wasm$/,
                type: 'webassembly/async'
            },
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env']
                    }
                }
            }
        ]
    },
    experiments: {
        asyncWebAssembly: true
    },
    resolve: {
        extensions: ['.js', '.wasm']
    },
    devServer: {
        static: {
            directory: path.join(__dirname, 'examples'),
        },
        compress: true,
        port: 8080,
        headers: {
            'Cross-Origin-Embedder-Policy': 'require-corp',
            'Cross-Origin-Opener-Policy': 'same-origin'
        }
    },
    optimization: {
        splitChunks: {
            chunks: 'all'
        }
    }
};
"#;

        fs::write(&webpack_path, webpack_content)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(webpack_path)
    }

    fn generate_example_html(&self) -> Result<PathBuf> {
        let html_path = self.output_dir.join("examples").join("index.html");

        let html_content = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciRS2 Neural Network Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress {
            width: 100%;
            height: 20px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar {
            height: 100%;
            background: #007bff;
            width: 0%;
            transition: width 0.3s ease;
        }
        .result {
            background: white;
            padding: 15px;
            border-radius: 4px;
            margin: 10px 0;
            border: 1px solid #ddd;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
            border-color: #f5c6cb;
        }
        .success {
            background: #d4edda;
            color: #155724;
            border-color: #c3e6cb;
        }
        .log {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            font-family: monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <h1>SciRS2 Neural Network WebAssembly Demo</h1>
    
    <div class="container">
        <h2>Model Status</h2>
        <div id="status">Not loaded</div>
        <div id="support-info"></div>
        
        <div>
            <button id="load-btn">Load Model</button>
            <button id="test-btn" disabled>Run Test Prediction</button>
            <button id="benchmark-btn" disabled>Run Benchmark</button>
        </div>
        
        <div class="progress" id="progress-container" style="display: none;">
            <div class="progress-bar" id="progress-bar"></div>
        </div>
    </div>

    <div class="container">
        <h2>Input Configuration</h2>
        <div>
            <label for="input-data">Input Data (comma-separated):</label><br>
            <input type="text" id="input-data" value="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" style="width: 100%; padding: 5px;">
        </div>
        <div style="margin-top: 10px;">
            <label for="batch-size">Batch Size:</label>
            <input type="number" id="batch-size" value="1" min="1" max="100" style="width: 80px; padding: 5px;">
        </div>
    </div>

    <div class="container">
        <h2>Results</h2>
        <div id="results"></div>
    </div>

    <div class="container">
        <h2>Performance Metrics</h2>
        <div id="metrics"></div>
    </div>

    <div class="container">
        <h2>Console Log</h2>
        <div id="log" class="log"></div>
        <button onclick="clearLog()">Clear Log</button>
    </div>

    <script type="module">
        import { SciRS2Model, loadModel, checkWasmSupport } from '../js/scirs2-model.mjs';

        let model = null;
        const logElement = document.getElementById('log');

        function log(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.style.color = type === 'error' ? 'red' : type === 'success' ? 'green' : 'black';
            logEntry.textContent = `[${timestamp}] ${message}`;
            logElement.appendChild(logEntry);
            logElement.scrollTop = logElement.scrollHeight;
            console.log(message);
        }

        function updateStatus(status, isError = false) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = status;
            statusEl.style.color = isError ? 'red' : 'green';
        }

        function updateProgress(percent) {
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = percent + '%';
        }

        function showResults(results, isError = false) {
            const resultsEl = document.getElementById('results');
            const resultDiv = document.createElement('div');
            resultDiv.className = `result ${isError ? 'error' : 'success'}`;
            resultDiv.innerHTML = isError ? 
                `<strong>Error:</strong> ${results}` : 
                `<strong>Prediction Result:</strong><br><pre>${JSON.stringify(results, null, 2)}</pre>`;
            resultsEl.appendChild(resultDiv);
        }

        function showMetrics(metrics) {
            const metricsEl = document.getElementById('metrics');
            metricsEl.innerHTML = `
                <div><strong>Inference Time:</strong> ${metrics.inferenceTime}ms</div>
                <div><strong>Memory Usage:</strong> ${metrics.memoryUsage} bytes</div>
                <div><strong>Throughput:</strong> ${metrics.throughput} inferences/sec</div>
            `;
        }

        window.clearLog = function() {
            logElement.innerHTML = '';
        };

        // Check WebAssembly support
        const support = checkWasmSupport();
        document.getElementById('support-info').innerHTML = `
            <strong>WebAssembly Support:</strong><br>
            Basic: ${support.basic ? '✅' : '❌'}<br>
            SIMD: ${support.simd ? '✅' : '❌'}<br>
            Threads: ${support.threads ? '✅' : '❌'}<br>
            Bulk Memory: ${support.bulkMemory ? '✅' : '❌'}
        `;

        // Load model
        document.getElementById('load-btn').addEventListener('click', async () => {
            try {
                updateStatus('Loading model...', false);
                document.getElementById('progress-container').style.display = 'block';
                log('Starting model load...');

                model = await loadModel('../wasm/model.wasm', {
                    onProgress: (percent) => {
                        updateProgress(percent);
                        log(`Loading progress: ${percent.toFixed(1)}%`);
                    },
                    useCache: true
                });

                updateStatus('Model loaded successfully!', false);
                document.getElementById('progress-container').style.display = 'none';
                document.getElementById('test-btn').disabled = false;
                document.getElementById('benchmark-btn').disabled = false;
                log('Model loaded and ready for inference', 'success');

                const modelInfo = await model.getModelInfo();
                log(`Model info: ${JSON.stringify(modelInfo)}`);
            } catch (error) {
                updateStatus(`Failed to load model: ${error.message}`, true);
                log(`Error loading model: ${error.message}`, 'error');
                document.getElementById('progress-container').style.display = 'none';
            }
        });

        // Test prediction
        document.getElementById('test-btn').addEventListener('click', async () => {
            if (!model) return;

            try {
                const inputText = document.getElementById('input-data').value;
                const inputData = inputText.split(',').map(x => parseFloat(x.trim()));
                const inputShape = [1, inputData.length];

                log(`Running prediction with input: [${inputData.join(', ')}]`);
                const startTime = performance.now();
                
                const result = await model.predict(inputData, inputShape);
                
                const endTime = performance.now();
                const inferenceTime = endTime - startTime;

                log(`Prediction completed in ${inferenceTime.toFixed(2)}ms`, 'success');
                showResults({
                    input: inputData,
                    output: Array.from(result),
                    shape: inputShape,
                    inferenceTime: inferenceTime.toFixed(2) + 'ms'
                });

                const modelInfo = await model.getModelInfo();
                showMetrics({
                    inferenceTime: inferenceTime.toFixed(2),
                    memoryUsage: modelInfo.memoryUsage,
                    throughput: (1000 / inferenceTime).toFixed(2)
                });
            } catch (error) {
                log(`Prediction error: ${error.message}`, 'error');
                showResults(error.message, true);
            }
        });

        // Benchmark
        document.getElementById('benchmark-btn').addEventListener('click', async () => {
            if (!model) return;

            try {
                const inputText = document.getElementById('input-data').value;
                const inputData = inputText.split(',').map(x => parseFloat(x.trim()));
                const inputShape = [1, inputData.length];
                const batchSize = parseInt(document.getElementById('batch-size').value);
                const iterations = 100;

                log(`Running benchmark: ${iterations} iterations, batch size: ${batchSize}`);
                
                const times = [];
                for (let i = 0; i < iterations; i++) {
                    const startTime = performance.now();
                    await model.predict(inputData, inputShape);
                    const endTime = performance.now();
                    times.push(endTime - startTime);

                    if (i % 10 === 0) {
                        log(`Benchmark progress: ${i + 1}/${iterations}`);
                    }
                }

                const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                const minTime = Math.min(...times);
                const maxTime = Math.max(...times);
                const throughput = 1000 / avgTime;

                log(`Benchmark completed!`, 'success');
                showResults({
                    iterations,
                    averageTime: avgTime.toFixed(2) + 'ms',
                    minTime: minTime.toFixed(2) + 'ms',
                    maxTime: maxTime.toFixed(2) + 'ms',
                    throughput: throughput.toFixed(2) + ' inferences/sec'
                });

                showMetrics({
                    inferenceTime: avgTime.toFixed(2),
                    memoryUsage: 'N/A',
                    throughput: throughput.toFixed(2)
                });
            } catch (error) {
                log(`Benchmark error: ${error.message}`, 'error');
                showResults(error.message, true);
            }
        });

        log('Demo page loaded. WebAssembly support: ' + (support.basic ? 'Available' : 'Not available'));
    </script>
</body>
</html>
"#;

        fs::write(&html_path, html_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(html_path)
    }

    fn generate_web_worker(&self) -> Result<PathBuf> {
        let worker_path = self.output_dir.join("js").join("scirs2-worker.js");

        let worker_content = r#"/**
 * SciRS2 Model Web Worker
 * 
 * This worker provides background neural network inference to avoid blocking
 * the main thread during computation-intensive operations.
 */

// Import the WASM module
importScripts('./scirs2-model.js');

let model = null;
let isInitialized = false;

// Message handler
self.onmessage = async function(e) {
    const { id, type, data } = e.data;
    
    try {
        switch (type) {
            case 'INITIALIZE':
                await initializeModel(data);
                postMessage({ id, type: 'INITIALIZE_SUCCESS' });
                break;
                
            case 'PREDICT':
                const result = await predict(data);
                postMessage({ id, type: 'PREDICT_SUCCESS', result });
                break;
                
            case 'PREDICT_BATCH':
                const batchResult = await predictBatch(data);
                postMessage({ id, type: 'PREDICT_BATCH_SUCCESS', result: batchResult });
                break;
                
            case 'GET_MODEL_INFO':
                const info = getModelInfo();
                postMessage({ id, type: 'MODEL_INFO_SUCCESS', result: info });
                break;
                
            case 'DISPOSE':
                dispose();
                postMessage({ id, type: 'DISPOSE_SUCCESS' });
                break;
                
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        postMessage({ 
            id, 
            type: 'ERROR', 
            error: error.message 
        });
    }
};

async function initializeModel({ wasmUrl, options = {} }) {
    if (isInitialized) {
        throw new Error('Model already initialized');
    }
    
    // Load WASM module
    const response = await fetch(wasmUrl);
    const wasmBytes = await response.arrayBuffer();
    
    // Initialize model
    model = new SciRS2.SciRS2Model();
    await model.initialize(wasmBytes, options);
    
    isInitialized = true;
    
    postMessage({
        type: 'PROGRESS',
        progress: 100,
        message: 'Model initialized successfully'
    });
}

async function predict({ inputData, inputShape, options = {} }) {
    if (!isInitialized) {
        throw new Error('Model not initialized');
    }
    
    const result = await model.predict(inputData, inputShape, options);
    
    // Convert Float32Array to regular array for serialization
    return {
        output: Array.from(result),
        inputShape,
        outputShape: [result.length] // Simplified
    };
}

async function predictBatch({ inputs, inputShape, options = {} }) {
    if (!isInitialized) {
        throw new Error('Model not initialized');
    }
    
    const results = [];
    const totalInputs = inputs.length;
    
    for (let i = 0; i < totalInputs; i++) {
        const result = await model.predict(inputs[i], inputShape, options);
        results.push(Array.from(result));
        
        // Report progress
        if (i % 10 === 0 || i === totalInputs - 1) {
            postMessage({
                type: 'PROGRESS',
                progress: ((i + 1) / totalInputs) * 100,
                message: `Processed ${i + 1}/${totalInputs} inputs`
            });
        }
    }
    
    return {
        outputs: results,
        inputShape,
        batchSize: totalInputs
    };
}

function getModelInfo() {
    if (!isInitialized) {
        throw new Error('Model not initialized');
    }
    
    return model.getModelInfo();
}

function dispose() {
    if (model) {
        model.dispose();
        model = null;
        isInitialized = false;
    }
}

// Error handler
self.onerror = function(error) {
    postMessage({
        type: 'ERROR',
        error: error.message
    });
};

// Unhandled promise rejection handler
self.onunhandledrejection = function(event) {
    postMessage({
        type: 'ERROR',
        error: event.reason.message || 'Unhandled promise rejection'
    });
};

// Signal that worker is ready
postMessage({ type: 'WORKER_READY' });
"#;

        fs::write(&worker_path, worker_content).map_err(|e| NeuralError::IOError(e.to_string()))?;

        Ok(worker_path)
    }

    fn generate_documentation(&self) -> Result<Vec<PathBuf>> {
        let mut docs = Vec::new();

        // Generate README
        let readme_path = self.output_dir.join("README.md");
        let readme_content = r#"# SciRS2 Neural Network WebAssembly

High-performance neural network inference in the browser using WebAssembly.

## Features

- 🚀 High-performance WASM execution
- 🧠 Support for various neural network architectures
- 📦 Easy to integrate with existing web applications
- 🔧 TypeScript support with full type definitions
- 🌐 Modern JavaScript with ES modules
- ⚡ SIMD acceleration when available
- 🔄 Web Workers for background processing
- 💾 Intelligent caching with Cache API
- 📊 Built-in performance monitoring

## Installation

### Via NPM

```bash
npm install @scirs2/neural-wasm
```

### Via CDN

```html
<script type="module">
  import {{ SciRS2Model }} from 'https://unpkg.com/@scirs2/neural-wasm/js/scirs2-model.mjs';
</script>
```

## Quick Start

### Basic Usage

```javascript
import {{ loadModel }} from '@scirs2/neural-wasm';

// Load the model
const model = await loadModel('./model.wasm');

// Prepare input data
const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
const inputShape = [1, 5];

// Run inference
const output = await model.predict(input, inputShape);
console.log('Prediction result:', output);
```

### TypeScript Usage

```typescript
import {{ SciRS2Model, loadModel }} from '@scirs2/neural-wasm';

const model: SciRS2Model = await loadModel('./model.wasm', {{
  useCache: true,
  onProgress: (progress) => console.log(`Loading: ${{progress}}%`)
}});

const result = await model.predict(inputData, inputShape);
```

### Web Worker Usage

```javascript
// Create worker
const worker = new Worker('./js/scirs2-worker.js');

// Initialize model in worker
worker.postMessage({{
  type: 'INITIALIZE',
  data: {{ wasmUrl: './model.wasm' }}
}});

// Run prediction in worker
worker.postMessage({{
  type: 'PREDICT',
  data: {{ inputData, inputShape }}
}});

worker.onmessage = (e) => {{
  if (e.data.type === 'PREDICT_SUCCESS') {{
    console.log('Result:', e.data.result);
  }}
}};
```

## API Reference

### SciRS2Model

The main class for neural network inference.

#### Methods

##### `initialize(wasmSource, options)`

Initialize the WASM module.

- `wasmSource`: String (URL) or ArrayBuffer containing the WASM module
- `options`: Initialization options
  - `initialMemory`: Initial memory pages (default: 256)
  - `maximumMemory`: Maximum memory pages (default: 1024)
  - `enableSIMD`: Enable SIMD acceleration (default: auto-detect)
  - `enableThreads`: Enable multi-threading (default: false)

##### `predict(inputData, inputShape, options)`

Run inference on input data.

- `inputData`: Float32Array or number array
- `inputShape`: Array of dimensions
- `options`: Prediction options
  - `timeout`: Timeout in milliseconds (default: 5000)
  - `useWorker`: Run in Web Worker (default: false)

Returns: Promise<Float32Array>

##### `predictBatch(inputs, inputShape, options)`

Run batch inference on multiple inputs.

- `inputs`: Array of input tensors
- `inputShape`: Shape of each input tensor
- `options`: Prediction options
  - `maxConcurrency`: Maximum concurrent predictions (default: 4)

Returns: Promise<Float32Array[]>

##### `getModelInfo()`

Get model metadata and information.

Returns: Promise<ModelInfo>

##### `dispose()`

Clean up resources and free memory.

### Utility Functions

##### `loadModel(modelUrl, options)`

Load a model with advanced options.

- `modelUrl`: URL to the WASM model file
- `options`: Loading options
  - `useCache`: Enable caching (default: true)
  - `onProgress`: Progress callback function
  - `timeout`: Loading timeout in milliseconds

##### `checkWasmSupport()`

Check WebAssembly feature support.

Returns: Object with support flags

## Performance Optimization

### SIMD Acceleration

The library automatically detects and uses SIMD instructions when available:

```javascript
const support = checkWasmSupport();
if (support.simd) {{
  console.log('SIMD acceleration available');
}}
```

### Caching

Models are automatically cached for faster subsequent loads:

```javascript
const model = await loadModel('./model.wasm', {{
  useCache: true, // Enable caching (default)
}});
```

### Memory Management

The library includes automatic memory management:

```javascript
// Memory is automatically cleaned up
const result = await model.predict(input, shape);

// Manual cleanup if needed
model.dispose();
```

## Browser Support

- ✅ Chrome 57+
- ✅ Firefox 52+
- ✅ Safari 11+
- ✅ Edge 16+

### Feature Support

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| WebAssembly | 57+ | 52+ | 11+ | 16+ |
| SIMD | 91+ | 89+ | 16.4+ | 91+ |
| Threads | 70+* | 79+* | ❌ | 79+* |

*Requires secure context (HTTPS) and special headers

## Examples

Check the `examples/` directory for complete examples:

- [Basic Usage](examples/basic.html)
- [Advanced Features](examples/advanced.html)
- [Web Worker Integration](examples/worker.html)
- [Performance Benchmarks](examples/benchmark.html)

## Building from Source

```bash
# Clone the repository
git clone https://github.com/scirs2/neural-wasm.git
cd neural-wasm

# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
"#
        .to_string();

        fs::write(&readme_path, readme_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        docs.push(readme_path);

        // Generate TypeScript declarations
        let dts_path = self.output_dir.join("ts").join("scirs2-model.d.ts");
        let dts_content = r#"// TypeScript declarations for SciRS2 Neural Network WebAssembly

export interface WasmSupport {
    basic: boolean;
    simd: boolean;
    threads: boolean;
    bulkMemory: boolean;
    multiValue?: boolean;
    referenceTypes?: boolean;
}

export interface ModelInfo {
    version: string;
    inputShape: number[];
    outputShape: number[];
    parameters: number;
    memoryUsage?: number;
}

export interface InitializationOptions {
    initialMemory?: number;
    maximumMemory?: number;
    enableSIMD?: boolean;
    enableThreads?: boolean;
}

export interface LoadingOptions extends InitializationOptions {
    useCache?: boolean;
    onProgress?: (progress: number) => void;
    timeout?: number;
}

export interface PredictionOptions {
    timeout?: number;
    useWorker?: boolean;
    batchSize?: number;
    maxConcurrency?: number;
}

export class SciRS2Model {
    constructor();
    
    initialize(wasmSource: string | ArrayBuffer, options?: InitializationOptions): Promise<void>;
    
    predict(
        inputData: Float32Array | number[], 
        inputShape: number[], 
        options?: PredictionOptions
    ): Promise<Float32Array>;
    
    predictBatch(
        inputs: (Float32Array | number[])[], 
        inputShape: number[], 
        options?: PredictionOptions
    ): Promise<Float32Array[]>;
    
    getModelInfo(): Promise<ModelInfo>;
    
    supportsBatchInference(): boolean;
    
    isReady(): boolean;
    
    get capabilities(): {
        batchInference: boolean;
        streaming: boolean;
        webgl: boolean;
        simd: boolean;
    };
    
    dispose(): void;
}

export function loadModel(modelUrl: string, options?: LoadingOptions): Promise<SciRS2Model>;

export function checkWasmSupport(): WasmSupport;
"#;

        fs::write(&dts_path, dts_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        docs.push(dts_path);

        Ok(docs)
    }

    fn calculate_bundle_info(
        &self,
        wasm_path: &Path,
        binding_paths: &[PathBuf],
    ) -> Result<BundleInfo> {
        let wasm_size = fs::metadata(wasm_path)
            .map_err(|e| NeuralError::IOError(e.to_string()))?
            .len() as usize;

        let mut js_size = 0;
        for binding_path in binding_paths {
            js_size += fs::metadata(binding_path)
                .map_err(|e| NeuralError::IOError(e.to_string()))?
                .len() as usize;
        }

        let total_size = wasm_size + js_size;
        let compression_ratio = 0.7; // Assume 70% compression with gzip
        let estimated_load_time_ms =
            (total_size as f64 * compression_ratio / 1024.0 / 1024.0 * 1000.0) as u64; // Rough estimate

        Ok(BundleInfo {
            total_size,
            wasm_size,
            js_size,
            compression_ratio,
            estimated_load_time_ms,
        })
    }
}

impl Default for WasmCompilationConfig {
    fn default() -> Self {
        Self {
            target_version: WasmVersion::SIMD,
            optimization_level: WasmOptimization {
                size_level: 2,
                speed_level: 2,
                lto: true,
                strip_debug: true,
                dead_code_elimination: true,
                inline_level: InlineLevel::Conservative,
            },
            features: WasmFeatures {
                simd: true,
                threads: false,
                bulk_memory: true,
                reference_types: false,
                exception_handling: false,
                tail_calls: false,
                multi_value: true,
                wasi: false,
            },
            memory_config: WasmMemoryConfig {
                initial_pages: 256,
                maximum_pages: Some(1024),
                shared: false,
                growth_strategy: MemoryGrowthStrategy::OnDemand,
                alignment: MemoryAlignment {
                    data_alignment: 8,
                    function_alignment: 16,
                    simd_alignment: 16,
                },
            },
            exports: WasmExports {
                functions: vec![WasmFunctionExport {
                    name: "predict".to_string(),
                    signature: WasmSignature {
                        params: vec![WasmType::I32, WasmType::I32, WasmType::I32],
                        returns: vec![WasmType::I32],
                    },
                    documentation: Some("Run neural network inference".to_string()),
                    performance_hints: vec![
                        PerformanceHint::Intensive,
                        PerformanceHint::SIMDFriendly,
                    ],
                }],
                memory: vec![],
                globals: vec![],
                tables: vec![],
            },
            imports: WasmImports {
                functions: vec![],
                memory: vec![],
                globals: vec![],
                tables: vec![],
            },
            debug_config: WasmDebugConfig {
                debug_symbols: false,
                source_maps: false,
                function_names: true,
                debug_assertions: false,
                profiling: ProfilingConfig {
                    function_profiling: false,
                    memory_profiling: false,
                    instruction_profiling: false,
                    output_format: ProfilingFormat::JSON,
                },
            },
        }
    }
}

impl Default for WebIntegrationConfig {
    fn default() -> Self {
        Self {
            bindings: WebBindingConfig {
                target_language: WebBindingLanguage::Both,
                module_system: ModuleSystem::ESModules,
                type_definitions: true,
                documentation: true,
                bundling: BundlingConfig {
                    enable: true,
                    format: BundleFormat::Single,
                    minify: true,
                    tree_shaking: true,
                    code_splitting: false,
                },
            },
            acceleration: WebAccelerationConfig {
                webgl: WebGLConfig {
                    enable: false,
                    version: 2,
                    shader_optimization: true,
                    texture_formats: vec![TextureFormat::RGBA32F, TextureFormat::R32F],
                },
                webgpu: WebGPUConfig {
                    enable: false,
                    compute_shaders: true,
                    memory_optimization: true,
                    pipeline_caching: true,
                },
                simd_js: true,
                parallel: ParallelConfig {
                    web_workers: true,
                    max_workers: Some(4),
                    shared_memory: false,
                    work_stealing: false,
                },
            },
            caching: CachingConfig {
                enable: true,
                strategy: CacheStrategy::LRU,
                storage: CacheStorage::CacheAPI,
                ttl_seconds: Some(3600), // 1 hour
                versioning: VersioningStrategy::Hash,
            },
            progressive_loading: ProgressiveLoadingConfig {
                enable: true,
                strategy: LoadingStrategy::Lazy,
                chunk_size: 1024 * 1024, // 1MB chunks
                preloading: PreloadingConfig {
                    enable: true,
                    percentage: 0.1, // Preload 10%
                    on_idle: true,
                    on_interaction: false,
                },
                streaming: true,
            },
            workers: WorkerConfig {
                enable: true,
                worker_type: WorkerType::Dedicated,
                messaging: MessagingStrategy::Transfer,
                pool: WorkerPoolConfig {
                    min_workers: 1,
                    max_workers: 4,
                    idle_timeout_ms: 30000, // 30 seconds
                    queue_size: 100,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Dense;
    use crate::models::sequential::Sequential;
    use rand::SeedableRng;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_wasm_compilation_config_default() {
        let config = WasmCompilationConfig::default();
        assert_eq!(config.target_version, WasmVersion::SIMD);
        assert_eq!(config.optimization_level.size_level, 2);
        assert!(config.features.simd);
        assert!(!config.features.threads);
    }

    #[test]
    fn test_web_integration_config_default() {
        let config = WebIntegrationConfig::default();
        assert_eq!(config.bindings.target_language, WebBindingLanguage::Both);
        assert_eq!(config.bindings.module_system, ModuleSystem::ESModules);
        assert!(config.caching.enable);
        assert_eq!(config.caching.strategy, CacheStrategy::LRU);
    }

    #[test]
    fn test_wasm_compiler_creation() {
        let temp_dir = TempDir::new().unwrap();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        let mut model: Sequential<f32> = Sequential::new();
        let dense = Dense::new(10, 1, Some("relu"), &mut rng).unwrap();
        model.add_layer(dense);

        let wasm_config = WasmCompilationConfig::default();
        let web_config = WebIntegrationConfig::default();
        let metadata = PackageMetadata {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Test".to_string(),
            license: "MIT".to_string(),
            platforms: vec!["wasm".to_string()],
            dependencies: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            runtime_requirements: crate::serving::RuntimeRequirements {
                min_memory_mb: 256,
                cpu_requirements: crate::serving::CpuRequirements {
                    min_cores: 1,
                    instruction_sets: Vec::new(),
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: Vec::new(),
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            checksum: "test".to_string(),
        };

        let compiler = WasmCompiler::new(
            model,
            wasm_config,
            web_config,
            metadata,
            temp_dir.path().to_path_buf(),
        );

        assert_eq!(compiler.config.target_version, WasmVersion::SIMD);
    }

    #[test]
    fn test_wasm_features() {
        let features = WasmFeatures {
            simd: true,
            threads: false,
            bulk_memory: true,
            reference_types: false,
            exception_handling: false,
            tail_calls: false,
            multi_value: true,
            wasi: false,
        };

        assert!(features.simd);
        assert!(!features.threads);
        assert!(features.bulk_memory);
        assert!(features.multi_value);
    }

    #[test]
    fn test_memory_config() {
        let memory_config = WasmMemoryConfig {
            initial_pages: 256,
            maximum_pages: Some(1024),
            shared: false,
            growth_strategy: MemoryGrowthStrategy::OnDemand,
            alignment: MemoryAlignment {
                data_alignment: 8,
                function_alignment: 16,
                simd_alignment: 16,
            },
        };

        assert_eq!(memory_config.initial_pages, 256);
        assert_eq!(memory_config.maximum_pages, Some(1024));
        assert!(!memory_config.shared);
        assert_eq!(
            memory_config.growth_strategy,
            MemoryGrowthStrategy::OnDemand
        );
    }

    #[test]
    fn test_bundle_info() {
        let bundle_info = BundleInfo {
            total_size: 1024 * 1024, // 1MB
            wasm_size: 512 * 1024,   // 512KB
            js_size: 512 * 1024,     // 512KB
            compression_ratio: 0.7,
            estimated_load_time_ms: 700, // ~700ms for 1MB at 1.4MB/s
        };

        assert_eq!(bundle_info.total_size, 1024 * 1024);
        assert_eq!(bundle_info.wasm_size, 512 * 1024);
        assert_eq!(bundle_info.js_size, 512 * 1024);
        assert_eq!(bundle_info.compression_ratio, 0.7);
    }
}
