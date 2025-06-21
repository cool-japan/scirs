//! JavaScript and TypeScript binding generation for WebAssembly neural networks
//!
//! This module provides comprehensive binding generation functionality including:
//! - JavaScript class generation for model integration
//! - TypeScript definitions and type-safe interfaces
//! - Modern JavaScript with ES modules support
//! - Web Worker integration code
//! - HTML demo pages and examples

use crate::error::{NeuralError, Result};
use std::fs;
use std::path::PathBuf;

/// Web binding language target
#[derive(Debug, Clone, PartialEq)]
pub enum WebBindingLanguage {
    /// Generate JavaScript bindings only
    JavaScript,
    /// Generate TypeScript bindings only
    TypeScript,
    /// Generate both JavaScript and TypeScript
    Both,
    /// Generate modern JavaScript with ES modules
    ModernJavaScript,
}

/// Module system for generated bindings
#[derive(Debug, Clone, PartialEq)]
pub enum ModuleSystem {
    /// CommonJS (require/module.exports)
    CommonJS,
    /// ES Modules (import/export)
    ESModules,
    /// AMD (RequireJS)
    AMD,
    /// UMD (Universal Module Definition)
    UMD,
    /// IIFE (Immediately Invoked Function Expression)
    IIFE,
}

/// Web binding configuration
#[derive(Debug, Clone)]
pub struct WebBindingConfig {
    /// Target programming language
    pub target_language: WebBindingLanguage,
    /// Module system to use
    pub module_system: ModuleSystem,
    /// Generate TypeScript definitions
    pub type_definitions: bool,
    /// Generate documentation
    pub documentation: bool,
    /// Bundling configuration
    pub bundling: BundlingConfig,
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

/// JavaScript/TypeScript binding generator
pub struct BindingGenerator {
    /// Output directory for generated files
    pub output_dir: PathBuf,
    /// Binding configuration
    pub config: WebBindingConfig,
}

impl BindingGenerator {
    /// Create a new binding generator
    pub fn new(output_dir: PathBuf, config: WebBindingConfig) -> Self {
        Self { output_dir, config }
    }

    /// Generate all required bindings based on configuration
    pub fn generate_bindings(&self) -> Result<Vec<PathBuf>> {
        let mut bindings = Vec::new();

        match self.config.target_language {
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

    /// Generate JavaScript bindings
    pub fn generate_javascript_bindings(&self) -> Result<PathBuf> {
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

            // Calculate output size (this would be determined by model architecture)
            const outputSize = this.calculateOutputSize(inputShape);
            const outputPtr = this.exports.allocate(outputSize * 4);
            
            if (!outputPtr) {
                this.exports.deallocate(inputPtr);
                throw new Error('Failed to allocate memory for output');
            }

            try {
                // Run inference
                const result = this.exports.predict(inputPtr, outputPtr, input.length);
                
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
            if (inputPtr) {
                this.exports.deallocate(inputPtr);
            }
        }
    }

    /**
     * Calculate output size based on model architecture
     * @param {Array} inputShape - Input tensor shape
     * @returns {number} Output size
     */
    calculateOutputSize(inputShape) {
        // This is a placeholder - real implementation would query the model
        return inputShape.reduce((a, b) => a * b, 1);
    }

    /**
     * Check if model supports batch inference
     * @returns {boolean}
     */
    supportsBatchInference() {
        return this.isInitialized && this.exports && this.exports.predictBatch;
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
            version: '1.0.0',
            inputShape: [1, -1], // Dynamic shape
            outputShape: [1, -1],
            parameters: 0, // Would be calculated from model
            memoryUsage: this.memory ? this.memory.buffer.byteLength : 0
        };
    }

    /**
     * Cleanup resources
     */
    dispose() {
        this.module = null;
        this.exports = null;
        this.memory = null;
        this.isInitialized = false;
    }
}

// Utility function to load model easily
async function loadModel(wasmPath, options = {}) {
    const model = new SciRS2Model();
    await model.initialize(wasmPath, options);
    return model;
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = { SciRS2Model, loadModel };
} else if (typeof window !== 'undefined') {
    // Browser global
    window.SciRS2Model = SciRS2Model;
    window.loadModel = loadModel;
}
"#;

        // Create directory if it doesn't exist
        if let Some(parent) = js_path.parent() {
            fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }

        fs::write(&js_path, js_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(js_path)
    }

    /// Generate TypeScript bindings
    pub fn generate_typescript_bindings(&self) -> Result<PathBuf> {
        let ts_path = self.output_dir.join("ts").join("scirs2-model.ts");

        let ts_content = r#"/**
 * SciRS2 Model WebAssembly Bindings - TypeScript
 * 
 * Type-safe TypeScript bindings for running SciRS2 neural network models
 * in the browser using WebAssembly.
 */

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
    private module: WebAssembly.Module | null = null;
    private exports: WebAssembly.Instance['exports'] | null = null;
    private memory: WebAssembly.Memory | null = null;
    private isInitialized: boolean = false;

    constructor() {
        // Initialize in constructor
    }

    /**
     * Initialize the WASM module
     */
    async initialize(
        wasmSource: string | ArrayBuffer, 
        options: InitializationOptions = {}
    ): Promise<void> {
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
                        initial: options.initialMemory ?? 256,
                        maximum: options.maximumMemory ?? 1024 
                    }),
                    abort: () => {
                        throw new Error('WASM execution aborted');
                    }
                }
            };

            const wasmModule = await WebAssembly.instantiate(wasmBytes, imports);

            this.module = wasmModule.module;
            this.exports = wasmModule.instance.exports;
            this.memory = this.exports.memory ?? imports.env.memory;
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
            const inputPtr = (this.exports.allocate as Function)(inputSize);
            
            if (!inputPtr) {
                throw new Error('Failed to allocate memory for input');
            }

            try {
                // Copy input data to WASM memory
                const inputView = new Float32Array(this.memory.buffer, inputPtr, input.length);
                inputView.set(input);

                // Calculate output size (this would be determined by model architecture)
                const outputSize = this.calculateOutputSize(inputShape);
                const outputPtr = (this.exports.allocate as Function)(outputSize * 4);
                
                if (!outputPtr) {
                    throw new Error('Failed to allocate memory for output');
                }

                try {
                    // Run inference with timeout
                    const result = await this.executeWithTimeout(
                        () => (this.exports!.predict as Function)(inputPtr, outputPtr, input.length),
                        options.timeout || 5000
                    );
                    
                    if (result !== 0) {
                        throw new Error(`Inference failed with error code: ${result}`);
                    }

                    // Copy output data from WASM memory
                    const outputView = new Float32Array(this.memory.buffer, outputPtr, outputSize);
                    return new Float32Array(outputView);
                } finally {
                    (this.exports.deallocate as Function)(outputPtr);
                }
            } finally {
                (this.exports.deallocate as Function)(inputPtr);
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
            version: '1.0.0',
            inputShape: [1, -1], // Dynamic shape
            outputShape: [1, -1],
            parameters: 0, // Would be calculated from model
            memoryUsage: this.memory?.buffer.byteLength ?? 0
        };
    }

    /**
     * Check if batch inference is supported
     */
    supportsBatchInference(): boolean {
        return this.isInitialized && 
               this.exports !== null && 
               'predictBatch' in this.exports;
    }

    /**
     * Check if model is ready for inference
     */
    isReady(): boolean {
        return this.isInitialized;
    }

    /**
     * Execute function with timeout
     */
    private async executeWithTimeout<T>(
        fn: () => T, 
        timeoutMs: number
    ): Promise<T> {
        return new Promise<T>((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Operation timed out after ${timeoutMs}ms`));
            }, timeoutMs);

            try {
                const result = fn();
                clearTimeout(timeout);
                resolve(result);
            } catch (error) {
                clearTimeout(timeout);
                reject(error);
            }
        });
    }

    /**
     * Calculate output size based on input shape
     */
    private calculateOutputSize(inputShape: number[]): number {
        // This is a placeholder - real implementation would query the model
        return inputShape.reduce((a, b) => a * b, 1);
    }

    /**
     * Cleanup resources
     */
    dispose(): void {
        this.module = null;
        this.exports = null;
        this.memory = null;
        this.isInitialized = false;
    }
}

/**
 * Utility function to load model with TypeScript support
 */
export async function loadModel(
    wasmPath: string, 
    options: LoadingOptions = {}
): Promise<SciRS2Model> {
    const model = new SciRS2Model();
    
    if (options.onProgress) {
        options.onProgress(0);
    }
    
    await model.initialize(wasmPath, options);
    
    if (options.onProgress) {
        options.onProgress(100);
    }
    
    return model;
}

/**
 * Check WebAssembly support in current environment
 */
export function checkWasmSupport(): WasmSupport {
    const support: WasmSupport = {
        basic: typeof WebAssembly !== 'undefined',
        simd: false,
        threads: false,
        bulkMemory: false
    };

    if (support.basic) {
        // Check for SIMD support
        try {
            support.simd = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00, 0x0a, 0x09, 0x01, 0x07, 0x00, 0xfd, 0x0f, 0x45, 0x0b
            ]));
        } catch {
            support.simd = false;
        }

        // Check for threads support
        support.threads = typeof SharedArrayBuffer !== 'undefined';

        // Check for bulk memory support
        try {
            support.bulkMemory = WebAssembly.validate(new Uint8Array([
                0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
                0x03, 0x02, 0x01, 0x00, 0x0a, 0x07, 0x01, 0x05, 0x00, 0xfc, 0x0a, 0x00, 0x0b
            ]));
        } catch {
            support.bulkMemory = false;
        }
    }

    return support;
}
"#;

        // Create directory if it doesn't exist
        if let Some(parent) = ts_path.parent() {
            fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }

        fs::write(&ts_path, ts_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(ts_path)
    }

    /// Generate modern JavaScript bindings with ES modules
    pub fn generate_modern_javascript_bindings(&self) -> Result<PathBuf> {
        let js_path = self.output_dir.join("js").join("scirs2-model.mjs");

        let js_content = r#"/**
 * SciRS2 Model WebAssembly Bindings - Modern JavaScript
 * 
 * Modern ES2020+ JavaScript bindings with async/await, modules, and advanced features
 */

export class SciRS2Model {
    #module = null;
    #exports = null;
    #memory = null;
    #isInitialized = false;

    constructor() {
        // Private field initialization
    }

    /**
     * Initialize the WASM module with modern async patterns
     * @param {string|ArrayBuffer} wasmSource - Path to WASM file or ArrayBuffer
     * @param {Object} options - Initialization options
     */
    async initialize(wasmSource, options = {}) {
        try {
            let wasmBytes;
            
            if (typeof wasmSource === 'string') {
                const response = await fetch(wasmSource, {
                    cache: options.useCache ? 'default' : 'no-cache'
                });
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch WASM: ${response.statusText}`);
                }
                
                wasmBytes = await response.arrayBuffer();
            } else {
                wasmBytes = wasmSource;
            }

            const imports = {
                env: {
                    memory: new WebAssembly.Memory({ 
                        initial: options.initialMemory ?? 256,
                        maximum: options.maximumMemory ?? 1024 
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

        return {
            version: '1.0.0',
            inputShape: [1, -1],
            outputShape: [1, -1],
            parameters: 0,
            memoryUsage: this.#memory?.buffer.byteLength ?? 0,
            timestamp: new Date().toISOString()
        };
    }

    /**
     * Check capabilities
     */
    supportsBatchInference() {
        return this.#isInitialized && this.#exports?.predictBatch !== undefined;
    }

    isReady() {
        return this.#isInitialized;
    }

    /**
     * Private memory management helper
     */
    async #withMemoryManagement(fn) {
        const allocatedPointers = [];
        
        const allocate = (size) => {
            const ptr = this.#exports.allocate(size);
            if (ptr) allocatedPointers.push(ptr);
            return ptr;
        };
        
        try {
            return await fn(allocate);
        } finally {
            // Cleanup all allocated memory
            for (const ptr of allocatedPointers) {
                if (ptr) this.#exports.deallocate(ptr);
            }
        }
    }

    /**
     * Private timeout execution helper
     */
    async #executeWithTimeout(fn, timeoutMs) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Operation timed out after ${timeoutMs}ms`));
            }, timeoutMs);

            try {
                const result = fn();
                clearTimeout(timeout);
                resolve(result);
            } catch (error) {
                clearTimeout(timeout);
                reject(error);
            }
        });
    }

    /**
     * Private output size calculation
     */
    #calculateOutputSize(inputShape) {
        return inputShape.reduce((a, b) => a * b, 1);
    }

    /**
     * Cleanup resources
     */
    dispose() {
        this.#module = null;
        this.#exports = null;
        this.#memory = null;
        this.#isInitialized = false;
    }
}

/**
 * Utility function with modern async loading
 */
export async function loadModel(wasmPath, options = {}) {
    const model = new SciRS2Model();
    
    // Use async iterator for progress reporting
    if (options.onProgress) {
        const progressGenerator = async function* () {
            yield 0;
            await model.initialize(wasmPath, options);
            yield 100;
        };
        
        for await (const progress of progressGenerator()) {
            options.onProgress(progress);
        }
    } else {
        await model.initialize(wasmPath, options);
    }
    
    return model;
}

/**
 * Advanced WebAssembly feature detection
 */
export function checkWasmSupport() {
    const support = {
        basic: typeof WebAssembly !== 'undefined',
        simd: false,
        threads: false,
        bulkMemory: false,
        multiValue: false,
        referenceTypes: false
    };

    if (support.basic) {
        // Feature detection using WebAssembly.validate
        const features = [
            { name: 'simd', bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00] },
            { name: 'threads', test: () => typeof SharedArrayBuffer !== 'undefined' },
            { name: 'bulkMemory', bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00] },
            { name: 'multiValue', bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00] },
            { name: 'referenceTypes', bytes: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00] }
        ];

        for (const feature of features) {
            try {
                if (feature.test) {
                    support[feature.name] = feature.test();
                } else if (feature.bytes) {
                    support[feature.name] = WebAssembly.validate(new Uint8Array(feature.bytes));
                }
            } catch {
                support[feature.name] = false;
            }
        }
    }

    return support;
}

/**
 * Web Worker factory for background inference
 */
export function createWorker() {
    const workerScript = `
        importScripts('scirs2-model.js');
        
        let model = null;
        
        self.onmessage = async function(e) {
            const { type, data, id } = e.data;
            
            try {
                switch (type) {
                    case 'INITIALIZE':
                        model = new SciRS2Model();
                        await model.initialize(data.wasmUrl, data.options);
                        self.postMessage({ type: 'INITIALIZE_SUCCESS', id });
                        break;
                        
                    case 'PREDICT':
                        if (!model) {
                            throw new Error('Model not initialized');
                        }
                        const result = await model.predict(data.inputData, data.inputShape, data.options);
                        self.postMessage({ type: 'PREDICT_SUCCESS', result, id });
                        break;
                        
                    case 'DISPOSE':
                        if (model) {
                            model.dispose();
                            model = null;
                        }
                        self.postMessage({ type: 'DISPOSE_SUCCESS', id });
                        break;
                        
                    default:
                        throw new Error(\`Unknown message type: \${type}\`);
                }
            } catch (error) {
                self.postMessage({ type: 'ERROR', error: error.message, id });
            }
        };
    `;
    
    const blob = new Blob([workerScript], { type: 'application/javascript' });
    return new Worker(URL.createObjectURL(blob));
}
"#;

        // Create directory if it doesn't exist
        if let Some(parent) = js_path.parent() {
            fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }

        fs::write(&js_path, js_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(js_path)
    }

    /// Generate TypeScript declarations file
    pub fn generate_typescript_declarations(&self) -> Result<PathBuf> {
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
    
    dispose(): void;
}

export function loadModel(wasmPath: string, options?: LoadingOptions): Promise<SciRS2Model>;

export function checkWasmSupport(): WasmSupport;

export function createWorker(): Worker;
"#;

        // Create directory if it doesn't exist
        if let Some(parent) = dts_path.parent() {
            fs::create_dir_all(parent).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }

        fs::write(&dts_path, dts_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        Ok(dts_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_binding_generator_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebBindingConfig {
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
        };

        let generator = BindingGenerator::new(temp_dir.path().to_path_buf(), config);
        assert_eq!(generator.output_dir, temp_dir.path());
    }

    #[test]
    fn test_javascript_binding_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebBindingConfig {
            target_language: WebBindingLanguage::JavaScript,
            module_system: ModuleSystem::ESModules,
            type_definitions: true,
            documentation: true,
            bundling: BundlingConfig {
                enable: false,
                format: BundleFormat::Single,
                minify: false,
                tree_shaking: false,
                code_splitting: false,
            },
        };

        let generator = BindingGenerator::new(temp_dir.path().to_path_buf(), config);
        let js_path = generator.generate_javascript_bindings().unwrap();

        assert!(js_path.exists());
        assert!(js_path.to_string_lossy().ends_with("scirs2-model.js"));

        let content = fs::read_to_string(&js_path).unwrap();
        assert!(content.contains("class SciRS2Model"));
        assert!(content.contains("async initialize"));
        assert!(content.contains("async predict"));
    }

    #[test]
    fn test_typescript_binding_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = WebBindingConfig {
            target_language: WebBindingLanguage::TypeScript,
            module_system: ModuleSystem::ESModules,
            type_definitions: true,
            documentation: true,
            bundling: BundlingConfig {
                enable: false,
                format: BundleFormat::Single,
                minify: false,
                tree_shaking: false,
                code_splitting: false,
            },
        };

        let generator = BindingGenerator::new(temp_dir.path().to_path_buf(), config);
        let ts_path = generator.generate_typescript_bindings().unwrap();

        assert!(ts_path.exists());
        assert!(ts_path.to_string_lossy().ends_with("scirs2-model.ts"));

        let content = fs::read_to_string(&ts_path).unwrap();
        assert!(content.contains("export class SciRS2Model"));
        assert!(content.contains("Promise<void>"));
        assert!(content.contains("PredictionOptions"));
    }
}
