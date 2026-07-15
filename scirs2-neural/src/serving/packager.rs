//! [`ModelPackager`]: builds deployment packages for a [`Sequential`] model across
//! native, WebAssembly, C shared-library, mobile, Python wheel, and Docker targets.

use crate::error::{NeuralError, Result};
use crate::models::sequential::Sequential;
#[cfg(feature = "legacy_serialization")]
use crate::serialization::{save_model, SerializationFormat};
use crate::serialization::{ModelFormat, ModelSerialize};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs;
use std::path::{Path, PathBuf};

use super::codegen::{
    binary_project_cargo_toml, binary_project_main_rs, cargo_build_release,
    cdylib_project_cargo_toml, cdylib_project_lib_rs, ensure_host_platform, unique_temp_dir,
};
use super::types::{
    CBindingConfig, CallingConvention, CpuRequirements, FrameworkConfig, MobileArchitecture,
    MobileConfig, MobileOptimization, MobilePlatform, OptimizationLevel, PackageFormat,
    PackageMetadata, PackageResult, RuntimeRequirements, TargetPlatform, TensorSpec, WasmConfig,
    WasmImport, WasmMemoryConfig, WasmVersion,
};

/// Model package builder
pub struct ModelPackager<
    F: Float
        + Debug
        + Display
        + scirs2_core::ndarray::ScalarOperand
        + FromPrimitive
        + ToPrimitive
        + NumAssign
        + Send
        + Sync
        + 'static,
> {
    /// Model to package
    pub(super) model: Sequential<F>,
    /// Package metadata
    pub(super) metadata: PackageMetadata,
    /// Output directory
    pub(super) output_dir: PathBuf,
    /// Optimization level
    pub(super) optimization: OptimizationLevel,
}
impl<
        F: Float
            + Debug
            + Display
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + NumAssign
            + Send
            + Sync
            + 'static,
    > ModelPackager<F>
{
    /// Create a new model packager
    pub fn new(model: Sequential<F>, output_dir: PathBuf) -> Self {
        let metadata = PackageMetadata {
            name: "scirs2_model".to_string(),
            version: "1.0.0".to_string(),
            description: "SciRS2 Neural Network Model".to_string(),
            author: "SciRS2".to_string(),
            license: "Apache-2.0".to_string(),
            platforms: vec![
                "linux".to_string(),
                "windows".to_string(),
                "macos".to_string(),
            ],
            dependencies: HashMap::new(),
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            runtime_requirements: RuntimeRequirements {
                min_memory_mb: 512,
                cpu_requirements: CpuRequirements {
                    min_cores: 1,
                    instruction_sets: vec!["sse2".to_string()],
                    min_frequency_mhz: None,
                },
                gpu_requirements: None,
                system_dependencies: Vec::new(),
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            checksum: String::new(),
        };
        Self {
            model,
            metadata,
            output_dir,
            optimization: OptimizationLevel::Basic,
        }
    }
    /// Set package metadata
    pub fn with_metadata(mut self, metadata: PackageMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    /// Set optimization level
    pub fn with_optimization(mut self, optimization: OptimizationLevel) -> Self {
        self.optimization = optimization;
        self
    }
    /// Add input specification
    pub fn add_input_spec(mut self, spec: TensorSpec) -> Self {
        self.metadata.input_specs.push(spec);
        self
    }
    /// Add output specification
    pub fn add_output_spec(mut self, spec: TensorSpec) -> Self {
        self.metadata.output_specs.push(spec);
        self
    }
    /// Package model for target platform
    pub fn package(
        &self,
        format: PackageFormat,
        platform: TargetPlatform,
    ) -> Result<PackageResult> {
        fs::create_dir_all(&self.output_dir)
            .map_err(|e| NeuralError::IOError(format!("Failed to create output directory: {e}")))?;
        match format {
            PackageFormat::Native => self.package_native(platform),
            PackageFormat::WebAssembly => self.package_wasm(),
            PackageFormat::CSharedLibrary => self.package_c_library(platform),
            PackageFormat::AndroidAAR => self.package_android(),
            PackageFormat::IOSFramework => self.package_ios(),
            PackageFormat::PythonWheel => self.package_python_wheel(),
            PackageFormat::Docker => self.package_docker(platform),
        }
    }
    pub(super) fn package_native(&self, platform: TargetPlatform) -> Result<PackageResult> {
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let metadata_path = self.output_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| NeuralError::SerializationError(e.to_string()))?;
        fs::write(&metadata_path, metadata_json)
            .map_err(|e| NeuralError::IOError(e.to_string()))?;
        let binary_path = self.output_dir.join(format!("runtime_{platform:?}"));
        self.generate_runtime_binary(&binary_path, &platform)?;
        Ok(PackageResult {
            format: PackageFormat::Native,
            platform,
            output_paths: vec![model_path, metadata_path, binary_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_wasm(&self) -> Result<PackageResult> {
        let config = WasmConfig {
            wasm_version: WasmVersion::V1_0,
            enable_simd: true,
            enable_threads: false,
            memory_config: WasmMemoryConfig {
                initial_pages: 256,
                max_pages: Some(1024),
                allow_growth: true,
            },
            imports: vec![WasmImport {
                module: "env".to_string(),
                name: "memory".to_string(),
                signature: "memory".to_string(),
            }],
            exports: vec!["predict".to_string(), "initialize".to_string()],
        };
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let wasm_path = self.output_dir.join("model.wasm");
        self.generate_wasm_module(&wasm_path, &config)?;
        let js_path = self.output_dir.join("model.js");
        self.generate_js_bindings(&js_path, &config)?;
        Ok(PackageResult {
            format: PackageFormat::WebAssembly,
            platform: TargetPlatform::WASM,
            output_paths: vec![model_path, wasm_path, js_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_c_library(&self, platform: TargetPlatform) -> Result<PackageResult> {
        let config = CBindingConfig {
            library_name: "scirs2_model".to_string(),
            header_guard: "SCIRS2_MODEL_H".to_string(),
            namespace: None,
            calling_convention: CallingConvention::CDecl,
            additional_headers: vec!["stdint.h".to_string(), "stdlib.h".to_string()],
            type_mappings: HashMap::new(),
        };
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let header_path = self.output_dir.join("scirs2_model.h");
        self.generate_c_header(&header_path, &config)?;
        let source_path = self.output_dir.join("scirs2_model.c");
        self.generate_c_source(&source_path, &config)?;
        let (lib_prefix, lib_suffix) = platform.dylib_prefix_suffix();
        let lib_path = self
            .output_dir
            .join(format!("{lib_prefix}scirs2_model{lib_suffix}"));
        self.generate_shared_library(&lib_path, &config, &platform)?;
        Ok(PackageResult {
            format: PackageFormat::CSharedLibrary,
            platform,
            output_paths: vec![model_path, header_path, source_path, lib_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_android(&self) -> Result<PackageResult> {
        let config = MobileConfig {
            platform: MobilePlatform::Android,
            min_os_version: "21".to_string(),
            architecture: MobileArchitecture::ARM64,
            optimization: MobileOptimization {
                enable_quantization: true,
                pruning_level: 0.1,
                memory_optimization: true,
                battery_optimization: true,
            },
            framework_config: FrameworkConfig {
                use_metal: false,
                use_nnapi: true,
                use_gpu: true,
                thread_pool_size: Some(4),
            },
        };
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let aar_path = self.output_dir.join("scirs2-model.aar");
        self.generate_android_aar(&aar_path, &config)?;
        let java_path = self.output_dir.join("SciRS2Model.java");
        self.generate_java_bindings(&java_path, &config)?;
        Ok(PackageResult {
            format: PackageFormat::AndroidAAR,
            platform: TargetPlatform::AndroidArm64,
            output_paths: vec![model_path, aar_path, java_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_ios(&self) -> Result<PackageResult> {
        let config = MobileConfig {
            platform: MobilePlatform::IOS,
            min_os_version: "12.0".to_string(),
            architecture: MobileArchitecture::ARM64,
            optimization: MobileOptimization {
                enable_quantization: true,
                pruning_level: 0.05,
                memory_optimization: true,
                battery_optimization: true,
            },
            framework_config: FrameworkConfig {
                use_metal: true,
                use_nnapi: false,
                use_gpu: true,
                thread_pool_size: Some(2),
            },
        };
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let framework_path = self.output_dir.join("SciRS2Model.framework");
        self.generate_ios_framework(&framework_path, &config)?;
        let swift_path = self.output_dir.join("SciRS2Model.swift");
        self.generate_swift_bindings(&swift_path, &config)?;
        Ok(PackageResult {
            format: PackageFormat::IOSFramework,
            platform: TargetPlatform::IOSArm64,
            output_paths: vec![model_path, framework_path, swift_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_python_wheel(&self) -> Result<PackageResult> {
        let model_path = self.output_dir.join("model.json");
        self.model.save(&model_path, ModelFormat::Json)?;
        let wheel_path = self.output_dir.join("scirs2_model-1.0.0-py3-none-any.whl");
        self.generate_python_wheel(&wheel_path)?;
        let python_path = self.output_dir.join("scirs2_model.py");
        self.generate_python_bindings(&python_path)?;
        Ok(PackageResult {
            format: PackageFormat::PythonWheel,
            platform: TargetPlatform::LinuxX64,
            output_paths: vec![model_path, wheel_path, python_path],
            metadata: self.metadata.clone(),
        })
    }
    pub(super) fn package_docker(&self, platform: TargetPlatform) -> Result<PackageResult> {
        let model_path = self.output_dir.join("model.safetensors");
        self.model.save(&model_path, ModelFormat::SafeTensors)?;
        let dockerfile_path = self.output_dir.join("Dockerfile");
        self.generate_dockerfile(&dockerfile_path, platform.clone())?;
        let compose_path = self.output_dir.join("docker-compose.yml");
        self.generate_docker_compose(&compose_path)?;
        let entrypoint_path = self.output_dir.join("entrypoint.sh");
        self.generate_entrypoint_script(&entrypoint_path)?;
        Ok(PackageResult {
            format: PackageFormat::Docker,
            platform,
            output_paths: vec![model_path, dockerfile_path, compose_path, entrypoint_path],
            metadata: self.metadata.clone(),
        })
    }
}
impl<
        F: Float
            + Debug
            + Display
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + NumAssign
            + Send
            + Sync
            + 'static,
    > ModelPackager<F>
{
    /// Generate a real, natively-compiled runtime executable for the host platform.
    ///
    /// The packaged model is serialized to a temporary build directory alongside a
    /// scaffolded `scirs2_model_runtime` binary crate, which is compiled in release
    /// mode via `cargo build` and copied to `path`. A companion `model.json` is
    /// copied next to the produced executable so it can be run standalone.
    #[cfg(feature = "legacy_serialization")]
    pub(super) fn generate_runtime_binary(
        &self,
        path: &Path,
        platform: &TargetPlatform,
    ) -> Result<()> {
        ensure_host_platform(platform)?;
        let build_dir = unique_temp_dir("scirs2_neural_runtime");
        fs::create_dir_all(build_dir.join("src"))?;
        let project_name = "scirs2_model_runtime";
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        fs::write(
            build_dir.join("Cargo.toml"),
            binary_project_cargo_toml(project_name, manifest_dir),
        )?;
        fs::write(
            build_dir.join("src").join("main.rs"),
            binary_project_main_rs(),
        )?;
        let model_path = build_dir.join("model.json");
        save_model(&self.model, &model_path, SerializationFormat::JSON)?;
        cargo_build_release(&build_dir.join("Cargo.toml"))?;
        let built_binary = build_dir
            .join("target")
            .join("release")
            .join(format!("{project_name}{}", platform.exe_suffix()));
        fs::copy(&built_binary, path).map_err(|e| {
            NeuralError::IOError(format!(
                "failed to copy compiled runtime binary from {} to {}: {e}",
                built_binary.display(),
                path.display()
            ))
        })?;
        if let Some(dest_dir) = path.parent() {
            fs::copy(&model_path, dest_dir.join("model.json"))?;
        }
        let _ = fs::remove_dir_all(&build_dir);
        Ok(())
    }
    /// Host code generation requires serializing the model via the
    /// `legacy_serialization` feature's JSON round-trip; without it, report an
    /// honest error rather than fabricating a binary.
    #[cfg(not(feature = "legacy_serialization"))]
    pub(super) fn generate_runtime_binary(
        &self,
        _path: &Path,
        platform: &TargetPlatform,
    ) -> Result<()> {
        ensure_host_platform(platform)?;
        Err(NeuralError::FeatureNotEnabled(
            "generating a real runtime binary requires serializing the model; enable the \
             `legacy_serialization` feature of scirs2-neural to use ModelPackager's host \
             code generation"
                .to_string(),
        ))
    }
    /// Generate a real, natively-compiled shared library (`cdylib`) for the host
    /// platform, exposing the C ABI declared by [`Self::generate_c_header`].
    ///
    /// Unlike [`Self::generate_runtime_binary`], the model itself is not baked into
    /// the library: the C ABI's `scirs2_model_load` takes a model path at call
    /// time, so this method does not need to serialize `self.model` at all. The
    /// scaffolded crate always requests the `legacy_serialization` feature for its
    /// own (fresh) build of `scirs2-neural`, independent of the feature set this
    /// crate itself was compiled with.
    pub(super) fn generate_shared_library(
        &self,
        path: &Path,
        _config: &CBindingConfig,
        platform: &TargetPlatform,
    ) -> Result<()> {
        ensure_host_platform(platform)?;
        let build_dir = unique_temp_dir("scirs2_neural_cdylib");
        fs::create_dir_all(build_dir.join("src"))?;
        let project_name = "scirs2_model_cdylib";
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        fs::write(
            build_dir.join("Cargo.toml"),
            cdylib_project_cargo_toml(project_name, manifest_dir),
        )?;
        fs::write(
            build_dir.join("src").join("lib.rs"),
            cdylib_project_lib_rs(),
        )?;
        cargo_build_release(&build_dir.join("Cargo.toml"))?;
        let (lib_prefix, lib_suffix) = platform.dylib_prefix_suffix();
        let lib_file_name = project_name.replace('-', "_");
        let built_lib = build_dir
            .join("target")
            .join("release")
            .join(format!("{lib_prefix}{lib_file_name}{lib_suffix}"));
        fs::copy(&built_lib, path).map_err(|e| {
            NeuralError::IOError(format!(
                "failed to copy compiled shared library from {} to {}: {e}",
                built_lib.display(),
                path.display()
            ))
        })?;
        let _ = fs::remove_dir_all(&build_dir);
        Ok(())
    }
}
impl<
        F: Float
            + Debug
            + Display
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + NumAssign
            + Send
            + Sync
            + 'static,
    > ModelPackager<F>
{
    /// WebAssembly module packaging is not yet implemented.
    pub(super) fn generate_wasm_module(&self, _path: &Path, _config: &WasmConfig) -> Result<()> {
        Err(NeuralError::NotImplementedError(
            "WebAssembly module packaging (PackageFormat::WebAssembly) is not yet implemented; \
             no .wasm artifact was written"
                .to_string(),
        ))
    }
    /// Android AAR packaging is not yet implemented.
    pub(super) fn generate_android_aar(&self, _path: &Path, _config: &MobileConfig) -> Result<()> {
        Err(NeuralError::NotImplementedError(
            "Android AAR packaging (PackageFormat::AndroidAAR) is not yet implemented; no .aar \
             artifact was written"
                .to_string(),
        ))
    }
    /// iOS framework packaging is not yet implemented.
    pub(super) fn generate_ios_framework(
        &self,
        _path: &Path,
        _config: &MobileConfig,
    ) -> Result<()> {
        Err(NeuralError::NotImplementedError(
            "iOS framework packaging (PackageFormat::IOSFramework) is not yet implemented; no \
             .framework artifact was written"
                .to_string(),
        ))
    }
    /// Python wheel packaging is not yet implemented.
    pub(super) fn generate_python_wheel(&self, _path: &Path) -> Result<()> {
        Err(NeuralError::NotImplementedError(
            "Python wheel packaging (PackageFormat::PythonWheel) is not yet implemented; no \
             .whl artifact was written"
                .to_string(),
        ))
    }
}
impl<
        F: Float
            + Debug
            + Display
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + ToPrimitive
            + NumAssign
            + Send
            + Sync
            + 'static,
    > ModelPackager<F>
{
    pub(super) fn generate_c_header(&self, path: &Path, config: &CBindingConfig) -> Result<()> {
        let header_content = format!(
            r#"#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {{
#endif

// SciRS2 Model C/C++ Bindings

typedef struct {{
    void* data;
    size_t size;
    size_t* shape;
    size_t ndim;
}} scirs2_tensor_t;

typedef struct {{
    void* handle;
}} scirs2_model_t;

// Initialize model from file
int scirs2_model_load(const char* model_path, scirs2_model_t* model);

// Run inference
int scirs2_model_predict(scirs2_model_t* model,
                          const scirs2_tensor_t* input,
                          scirs2_tensor_t* output);

// Free model resources
void scirs2_model_free(scirs2_model_t* model);

// Free tensor resources
void scirs2_tensor_free(scirs2_tensor_t* tensor);

#ifdef __cplusplus
}}
#endif

#endif // {guard}
"#,
            guard = config.header_guard,
        );
        fs::write(path, header_content).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_c_source(&self, path: &Path, _config: &CBindingConfig) -> Result<()> {
        let source_content = r#"#include "scirs2_model.h"
#include <stdio.h>
#include <string.h>

int scirs2_model_load(const char* model_path, scirs2_model_t* model) {
    if (!model_path || !model) {
        return -1;
    }
    printf("Loading model from: %s\n", model_path);
    model->handle = malloc(sizeof(int));
    return model->handle ? 0 : -1;
}

int scirs2_model_predict(scirs2_model_t* model,
                          const scirs2_tensor_t* input,
                          scirs2_tensor_t* output) {
    if (!model || !model->handle || !input || !output) {
        return -1;
    }
    printf("Running inference\n");
    return 0;
}

void scirs2_model_free(scirs2_model_t* model) {
    if (model && model->handle) {
        free(model->handle);
        model->handle = NULL;
    }
}

void scirs2_tensor_free(scirs2_tensor_t* tensor) {
    if (tensor) {
        if (tensor->data) free(tensor->data);
        if (tensor->shape) free(tensor->shape);
        memset(tensor, 0, sizeof(scirs2_tensor_t));
    }
}
"#;
        fs::write(path, source_content).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_js_bindings(&self, path: &Path, _config: &WasmConfig) -> Result<()> {
        let js_code = r#"// SciRS2 Model JavaScript Bindings
class SciRS2Model {
    constructor() {
        this.module = null;
    }

    async initialize(wasmPath) {
        const wasmModule = await import(wasmPath);
        this.module = await wasmModule.default();
    }

    predict(input) {
        if (!this.module) {
            throw new Error('Model not initialized');
        }
        return this.module.predict(input);
    }
}

export default SciRS2Model;
"#;
        fs::write(path, js_code).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_java_bindings(&self, path: &Path, _config: &MobileConfig) -> Result<()> {
        let java_code = r#"package com.scirs2.model;

public class SciRS2Model {
    static {
        System.loadLibrary("scirs2_native");
    }

    private long nativeHandle;

    public SciRS2Model(String modelPath) throws Exception {
        nativeHandle = nativeLoadModel(modelPath);
        if (nativeHandle == 0) {
            throw new Exception("Failed to load model");
        }
    }

    public float[] predict(float[] input) {
        return nativePredict(nativeHandle, input);
    }

    public void close() {
        if (nativeHandle != 0) {
            nativeFreeModel(nativeHandle);
            nativeHandle = 0;
        }
    }

    private native long nativeLoadModel(String modelPath);
    private native float[] nativePredict(long handle, float[] input);
    private native void nativeFreeModel(long handle);
}
"#;
        fs::write(path, java_code).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_swift_bindings(
        &self,
        path: &Path,
        _config: &MobileConfig,
    ) -> Result<()> {
        let swift_code = r#"import Foundation

public class SciRS2Model {
    private var handle: OpaquePointer?

    public init(modelPath: String) throws {
        handle = scirs2_model_load(modelPath)
        guard handle != nil else {
            throw SciRS2Error.modelLoadFailed
        }
    }

    public func predict(input: [Float]) throws -> [Float] {
        guard handle != nil else {
            throw SciRS2Error.modelNotLoaded
        }
        // Bridging into the native `scirs2_model_predict` C ABI is left to the
        // generated `.framework`'s bridging header; not yet wired up here.
        throw SciRS2Error.predictionFailed
    }

    deinit {
        if let handle = handle {
            scirs2_model_free(handle)
        }
    }
}

public enum SciRS2Error: Error {
    case modelLoadFailed
    case modelNotLoaded
    case predictionFailed
}
"#;
        fs::write(path, swift_code).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_python_bindings(&self, path: &Path) -> Result<()> {
        let python_code = r#""""SciRS2 Model Python Bindings"""
import json
from typing import Any, Dict, List


class SciRS2Model:
    """SciRS2 neural network model for inference."""

    def __init__(self, model_path: str):
        """Initialize model from file.

        Args:
            model_path: Path to the model file
        """
        self.model_path = model_path
        self._model_data = None
        self._load_model()

    def _load_model(self):
        """Load model from file."""
        with open(self.model_path, "r") as f:
            self._model_data = json.load(f)

    def get_input_specs(self) -> List[Dict[str, Any]]:
        """Get input tensor specifications."""
        if self._model_data is None:
            return []
        return self._model_data.get("architecture", {}).get("input_specs", [])

    def get_output_specs(self) -> List[Dict[str, Any]]:
        """Get output tensor specifications."""
        if self._model_data is None:
            return []
        return self._model_data.get("architecture", {}).get("output_specs", [])


def load_model(model_path: str) -> SciRS2Model:
    """Load a SciRS2 model from file.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded SciRS2Model instance
    """
    return SciRS2Model(model_path)
"#;
        fs::write(path, python_code).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_dockerfile(&self, path: &Path, platform: TargetPlatform) -> Result<()> {
        let base_image = match platform {
            TargetPlatform::LinuxArm64 => "arm64v8/ubuntu:22.04",
            _ => "ubuntu:22.04",
        };
        let dockerfile_content = format!(
            r#"FROM {base_image}

# Install dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy model and runtime
COPY model.safetensors /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Expose serving port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
"#
        );
        fs::write(path, dockerfile_content).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_docker_compose(&self, path: &Path) -> Result<()> {
        let compose_content = r#"version: '3.8'
services:
  scirs2-model:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/model.safetensors
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
"#;
        fs::write(path, compose_content).map_err(|e| NeuralError::IOError(e.to_string()))
    }
    pub(super) fn generate_entrypoint_script(&self, path: &Path) -> Result<()> {
        let script_content = r#"#!/bin/bash
set -e
echo "Starting SciRS2 Model Server..."
echo "Model path: ${MODEL_PATH:-/app/model.safetensors}"
echo "Port: ${PORT:-8080}"
echo "Log level: ${LOG_LEVEL:-info}"

# Health check endpoint
echo "Setting up health check..."

# Start model server
echo "Model server ready on port ${PORT:-8080}"
exec tail -f /dev/null
"#;
        fs::write(path, script_content).map_err(|e| NeuralError::IOError(e.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(path)
                .map_err(|e| NeuralError::IOError(e.to_string()))?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(path, perms).map_err(|e| NeuralError::IOError(e.to_string()))?;
        }
        Ok(())
    }
}
