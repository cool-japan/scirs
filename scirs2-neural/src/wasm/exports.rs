//! WebAssembly compilation and export configuration
//!
//! This module provides comprehensive WebAssembly compilation functionality including:
//! - WASM compilation configuration and optimization
//! - Export and import specifications
//! - Web acceleration configuration (WebGL, WebGPU)
//! - Debugging and profiling configuration
//! - Main WasmCompiler implementation

use super::bindings::{BindingGenerator, WebBindingConfig};
use super::memory::{CachingConfig, ParallelConfig, ProgressiveLoadingConfig, WasmMemoryConfig};
use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
use crate::serving::PackageMetadata;
use num_traits::Float;
use std::fmt::Debug;
use std::fs;
use std::path::{Path, PathBuf};

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
    /// Generate source maps
    pub source_maps: bool,
    /// Preserve function names
    pub function_names: bool,
    /// Enable debug assertions
    pub debug_assertions: bool,
    /// Profiling configuration
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
    /// JSON format
    JSON,
    /// Binary format
    Binary,
    /// Text format
    Text,
    /// Chrome DevTools format
    ChromeDevTools,
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

/// Web integration configuration
#[derive(Debug, Clone)]
pub struct WebIntegrationConfig {
    /// Web binding configuration
    pub bindings: WebBindingConfig,
    /// Acceleration configuration
    pub acceleration: WebAccelerationConfig,
    /// Caching configuration
    pub caching: CachingConfig,
    /// Progressive loading configuration
    pub progressive_loading: ProgressiveLoadingConfig,
    /// Worker configuration
    pub workers: WorkerConfig,
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
        let binding_generator =
            BindingGenerator::new(self.output_dir.clone(), self.web_config.bindings.clone());

        binding_generator.generate_bindings()
    }

    fn generate_web_files(&self) -> Result<Vec<PathBuf>> {
        let mut web_files = Vec::new();

        // Generate HTML demo if enabled
        if self.web_config.bindings.documentation {
            let html_path = self.generate_html_demo()?;
            web_files.push(html_path);
        }

        // Generate service worker if enabled
        if self.web_config.workers.enable
            && self.web_config.workers.worker_type == WorkerType::Service
        {
            let sw_path = self.generate_service_worker()?;
            web_files.push(sw_path);
        }

        // Generate WebGL shaders if enabled
        if self.web_config.acceleration.webgl.enable {
            let shader_files = self.generate_webgl_shaders()?;
            web_files.extend(shader_files);
        }

        Ok(web_files)
    }

    fn generate_html_demo(&self) -> Result<PathBuf> {
        let html_path = self.output_dir.join("examples").join("demo.html");

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
        <h2>Results</h2>
        <div id="results"></div>
    </div>

    <div class="container">
        <h2>Console Log</h2>
        <div id="log" class="log"></div>
    </div>

    <script type="module">
        import { loadModel, checkWasmSupport } from '../js/scirs2-model.mjs';

        let model = null;
        const statusEl = document.getElementById('status');
        const supportInfoEl = document.getElementById('support-info');
        const loadBtn = document.getElementById('load-btn');
        const testBtn = document.getElementById('test-btn');
        const benchmarkBtn = document.getElementById('benchmark-btn');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const resultsEl = document.getElementById('results');
        const logEl = document.getElementById('log');

        // Check WebAssembly support
        const support = checkWasmSupport();
        supportInfoEl.innerHTML = `
            <p>WebAssembly Support:</p>
            <ul>
                <li>Basic: ${support.basic ? '✅' : '❌'}</li>
                <li>SIMD: ${support.simd ? '✅' : '❌'}</li>
                <li>Threads: ${support.threads ? '✅' : '❌'}</li>
                <li>Bulk Memory: ${support.bulkMemory ? '✅' : '❌'}</li>
            </ul>
        `;

        function log(message) {
            const timestamp = new Date().toLocaleTimeString();
            logEl.innerHTML += `[${timestamp}] ${message}\n`;
            logEl.scrollTop = logEl.scrollHeight;
        }

        function showResult(message, type = 'success') {
            const resultDiv = document.createElement('div');
            resultDiv.className = `result ${type}`;
            resultDiv.textContent = message;
            resultsEl.appendChild(resultDiv);
        }

        function updateProgress(percent) {
            progressBar.style.width = `${percent}%`;
        }

        loadBtn.addEventListener('click', async () => {
            try {
                statusEl.textContent = 'Loading...';
                progressContainer.style.display = 'block';
                loadBtn.disabled = true;

                log('Starting model load...');
                
                model = await loadModel('../wasm/model.wasm', {
                    onProgress: updateProgress,
                    useCache: true
                });

                statusEl.textContent = 'Model loaded successfully';
                testBtn.disabled = false;
                benchmarkBtn.disabled = false;
                progressContainer.style.display = 'none';
                
                const info = await model.getModelInfo();
                log(`Model loaded: ${JSON.stringify(info)}`);
                showResult('Model loaded successfully!');

            } catch (error) {
                statusEl.textContent = 'Load failed';
                log(`Load error: ${error.message}`);
                showResult(`Load failed: ${error.message}`, 'error');
                loadBtn.disabled = false;
                progressContainer.style.display = 'none';
            }
        });

        testBtn.addEventListener('click', async () => {
            try {
                log('Running test prediction...');
                const input = new Float32Array([1, 2, 3, 4, 5]);
                const shape = [1, 5];
                
                const start = performance.now();
                const output = await model.predict(input, shape);
                const duration = performance.now() - start;
                
                log(`Prediction completed in ${duration.toFixed(2)}ms`);
                log(`Input: [${Array.from(input).join(', ')}]`);
                log(`Output: [${Array.from(output).join(', ')}]`);
                
                showResult(`Test prediction completed in ${duration.toFixed(2)}ms`);
            } catch (error) {
                log(`Prediction error: ${error.message}`);
                showResult(`Prediction failed: ${error.message}`, 'error');
            }
        });

        benchmarkBtn.addEventListener('click', async () => {
            try {
                log('Running benchmark...');
                const input = new Float32Array(100).fill(1);
                const shape = [1, 100];
                const iterations = 100;
                
                const times = [];
                for (let i = 0; i < iterations; i++) {
                    const start = performance.now();
                    await model.predict(input, shape);
                    times.push(performance.now() - start);
                    
                    if (i % 10 === 0) {
                        updateProgress((i / iterations) * 100);
                    }
                }
                
                const avgTime = times.reduce((a, b) => a + b) / times.length;
                const minTime = Math.min(...times);
                const maxTime = Math.max(...times);
                
                log(`Benchmark completed: ${iterations} iterations`);
                log(`Average: ${avgTime.toFixed(2)}ms`);
                log(`Min: ${minTime.toFixed(2)}ms, Max: ${maxTime.toFixed(2)}ms`);
                
                showResult(`Benchmark: ${avgTime.toFixed(2)}ms average (${iterations} runs)`);
                progressContainer.style.display = 'none';
                
            } catch (error) {
                log(`Benchmark error: ${error.message}`);
                showResult(`Benchmark failed: ${error.message}`, 'error');
                progressContainer.style.display = 'none';
            }
        });

        log('Demo page loaded');
    </script>
</body>
</html>"#;

        fs::write(&html_path, html_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(html_path)
    }

    fn generate_service_worker(&self) -> Result<PathBuf> {
        let sw_path = self.output_dir.join("dist").join("sw.js");

        let sw_content = r#"// SciRS2 Service Worker for model caching
const CACHE_NAME = 'scirs2-model-v1';
const MODEL_URLS = [
    './wasm/model.wasm',
    './js/scirs2-model.js',
    './js/scirs2-model.mjs'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(MODEL_URLS))
    );
});

self.addEventListener('fetch', (event) => {
    if (MODEL_URLS.some(url => event.request.url.includes(url))) {
        event.respondWith(
            caches.match(event.request)
                .then(response => response || fetch(event.request))
        );
    }
});
"#;

        fs::write(&sw_path, sw_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(sw_path)
    }

    fn generate_webgl_shaders(&self) -> Result<Vec<PathBuf>> {
        let mut shader_files = Vec::new();
        let shader_dir = self.output_dir.join("shaders");
        fs::create_dir_all(&shader_dir).map_err(|e| NeuralError::IOError(e.to_string()))?;

        // Vertex shader
        let vs_path = shader_dir.join("neural.vert");
        let vs_content = r#"#version 300 es
precision highp float;

in vec4 a_position;
in vec2 a_texCoord;

out vec2 v_texCoord;

void main() {
    gl_Position = a_position;
    v_texCoord = a_texCoord;
}
"#;
        fs::write(&vs_path, vs_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        shader_files.push(vs_path);

        // Fragment shader for neural network computation
        let fs_path = shader_dir.join("neural.frag");
        let fs_content = r#"#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_weights;
uniform sampler2D u_input;
uniform vec2 u_inputSize;
uniform vec2 u_outputSize;

void main() {
    vec2 outputCoord = v_texCoord;
    vec4 result = vec4(0.0);
    
    // Simple matrix multiplication in shader
    for (float i = 0.0; i < u_inputSize.x; i += 1.0) {
        vec2 inputCoord = vec2(i / u_inputSize.x, 0.0);
        vec2 weightCoord = vec2(i / u_inputSize.x, outputCoord.y);
        
        vec4 input_val = texture(u_input, inputCoord);
        vec4 weight_val = texture(u_weights, weightCoord);
        
        result += input_val * weight_val;
    }
    
    // Apply activation (ReLU)
    result = max(result, 0.0);
    
    fragColor = result;
}
"#;
        fs::write(&fs_path, fs_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        shader_files.push(fs_path);

        Ok(shader_files)
    }

    fn generate_documentation(&self) -> Result<Vec<PathBuf>> {
        let mut docs = Vec::new();

        // Generate README
        let readme_path = self.output_dir.join("README.md");
        let readme_content = format!(
            r#"# SciRS2 WebAssembly Neural Network

Generated WebAssembly bindings for SciRS2 neural network models.

## Features

- 🚀 High-performance WebAssembly execution
- 📱 Cross-platform browser support
- 🔧 TypeScript and JavaScript bindings
- 🎯 Optimized for neural network inference
- 🌐 Web Worker support for background processing
- 💾 Intelligent caching and progressive loading
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

## Configuration

This model was compiled with the following configuration:

- **Target Version**: {:?}
- **Optimization Level**: Size={}, Speed={}
- **Features**: SIMD={}, Threads={}, Bulk Memory={}
- **Memory**: {}KB initial, {}KB maximum

## Files

- `wasm/model.wasm` - The WebAssembly module
- `js/scirs2-model.js` - JavaScript bindings
- `js/scirs2-model.mjs` - ES module bindings  
- `ts/scirs2-model.ts` - TypeScript bindings
- `examples/demo.html` - Interactive demo

## License

MIT License - see [LICENSE](LICENSE) file for details.
"#,
            self.config.target_version,
            self.config.optimization_level.size_level,
            self.config.optimization_level.speed_level,
            self.config.features.simd,
            self.config.features.threads,
            self.config.features.bulk_memory,
            self.config.memory_config.initial_pages * 64, // Convert pages to KB
            self.config
                .memory_config
                .maximum_pages
                .map_or("unlimited".to_string(), |p| format!("{}KB", p * 64))
        );

        fs::write(&readme_path, readme_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        docs.push(readme_path);

        Ok(docs)
    }

    fn calculate_bundle_info(
        &self,
        wasm_module: &Path,
        bindings: &[PathBuf],
    ) -> Result<BundleInfo> {
        let wasm_size = fs::metadata(wasm_module)
            .map_err(|e| NeuralError::IOError(e.to_string()))?
            .len() as usize;

        let js_size = bindings
            .iter()
            .map(|path| {
                fs::metadata(path)
                    .map(|meta| meta.len() as usize)
                    .unwrap_or(0)
            })
            .sum();

        let total_size = wasm_size + js_size;
        let compression_ratio = 0.7; // Estimated compression ratio
        let estimated_load_time_ms = (total_size / 1024) as u64; // ~1ms per KB estimate

        Ok(BundleInfo {
            total_size,
            wasm_size,
            js_size,
            compression_ratio,
            estimated_load_time_ms,
        })
    }
}

// Default implementations
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
            memory_config: WasmMemoryConfig::default(),
            exports: WasmExports {
                functions: vec![],
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
        use super::bindings::{BundleFormat, BundlingConfig, ModuleSystem, WebBindingLanguage};

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
                parallel: ParallelConfig::default(),
            },
            caching: CachingConfig::default(),
            progressive_loading: ProgressiveLoadingConfig::default(),
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
        assert!(config.caching.enable);
        assert_eq!(config.workers.pool.max_workers, 4);
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
}
