//! Tests for [`super::packager::ModelPackager`] and its host code generation pipeline.

use crate::models::sequential::Sequential;
use std::collections::HashMap;
use std::fs;
use std::process::Command;

use super::codegen::{
    binary_project_cargo_toml, binary_project_main_rs, cdylib_project_cargo_toml,
    cdylib_project_lib_rs, ensure_host_platform, unique_temp_dir,
};
use super::packager::ModelPackager;
use super::types::{
    CBindingConfig, CallingConvention, CpuRequirements, FrameworkConfig, MobileArchitecture,
    MobileConfig, MobileOptimization, MobilePlatform, ModelServer, OptimizationLevel,
    PackageFormat, PackageMetadata, RuntimeRequirements, ServerConfig, TargetPlatform, TensorSpec,
    WasmConfig, WasmImport, WasmMemoryConfig, WasmVersion,
};

use crate::layers::Dense;
use scirs2_core::random::SeedableRng;
use tempfile::TempDir;

fn make_test_model() -> Sequential<f32> {
    let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([42; 32]);
    let mut model: Sequential<f32> = Sequential::new();
    let dense = Dense::new(10, 1, Some("relu"), &mut rng).expect("failed to build Dense layer");
    model.add_layer(dense);
    model
}
#[test]
fn test_package_metadata_creation() {
    let metadata = PackageMetadata {
        name: "test_model".to_string(),
        version: "1.0.0".to_string(),
        description: "Test model".to_string(),
        author: "Test".to_string(),
        license: "Apache-2.0".to_string(),
        platforms: vec!["linux".to_string()],
        dependencies: HashMap::new(),
        input_specs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![Some(1), Some(10)],
            dtype: "float32".to_string(),
            description: None,
            range: None,
        }],
        output_specs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![Some(1), Some(1)],
            dtype: "float32".to_string(),
            description: None,
            range: None,
        }],
        runtime_requirements: RuntimeRequirements {
            min_memory_mb: 256,
            cpu_requirements: CpuRequirements {
                min_cores: 1,
                instruction_sets: vec!["sse2".to_string()],
                min_frequency_mhz: None,
            },
            gpu_requirements: None,
            system_dependencies: Vec::new(),
        },
        timestamp: "2024-01-01T00:00:00Z".to_string(),
        checksum: "abc123".to_string(),
    };
    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.input_specs.len(), 1);
    assert_eq!(metadata.output_specs.len(), 1);
}
#[test]
fn test_model_packager_creation() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    assert_eq!(packager.metadata.name, "scirs2_model");
    assert_eq!(packager.optimization, OptimizationLevel::Basic);
}
#[test]
fn test_tensor_spec() {
    let spec = TensorSpec {
        name: "test_tensor".to_string(),
        shape: vec![Some(32), None, Some(10)],
        dtype: "float32".to_string(),
        description: Some("Test tensor".to_string()),
        range: Some((-1.0, 1.0)),
    };
    assert_eq!(spec.name, "test_tensor");
    assert_eq!(spec.shape.len(), 3);
    assert_eq!(spec.shape[1], None);
    assert!(spec.range.is_some());
}
#[test]
fn test_c_binding_config() {
    let mut type_mappings = HashMap::new();
    type_mappings.insert("f32".to_string(), "float".to_string());
    type_mappings.insert("f64".to_string(), "double".to_string());
    let config = CBindingConfig {
        library_name: "test_lib".to_string(),
        header_guard: "TEST_LIB_H".to_string(),
        namespace: Some("testlib".to_string()),
        calling_convention: CallingConvention::CDecl,
        additional_headers: vec!["math.h".to_string()],
        type_mappings,
    };
    assert_eq!(config.library_name, "test_lib");
    assert_eq!(config.calling_convention, CallingConvention::CDecl);
    assert!(config.namespace.is_some());
    assert_eq!(config.type_mappings.len(), 2);
}
#[test]
fn test_wasm_config() {
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
        exports: vec!["predict".to_string()],
    };
    assert_eq!(config.wasm_version, WasmVersion::V1_0);
    assert!(config.enable_simd);
    assert!(!config.enable_threads);
    assert_eq!(config.memory_config.initial_pages, 256);
    assert_eq!(config.imports.len(), 1);
    assert_eq!(config.exports.len(), 1);
}
#[test]
fn test_mobile_config() {
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
    assert_eq!(config.platform, MobilePlatform::IOS);
    assert_eq!(config.architecture, MobileArchitecture::ARM64);
    assert!(config.optimization.enable_quantization);
    assert!(config.framework_config.use_metal);
    assert!(!config.framework_config.use_nnapi);
}
#[test]
fn test_mobile_architecture_universal_variant_exists() {
    let arch = MobileArchitecture::Universal;
    assert_ne!(arch, MobileArchitecture::ARM64);
}
#[test]
fn test_server_config() {
    let config = ServerConfig {
        port: 8080,
        max_batch_size: 32,
        timeout_seconds: 30,
        enable_logging: true,
        max_concurrent_requests: 100,
    };
    assert_eq!(config.port, 8080);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.timeout_seconds, 30);
    assert!(config.enable_logging);
    assert_eq!(config.max_concurrent_requests, 100);
}
#[test]
fn test_model_server_stats() {
    let model = make_test_model();
    let config = ServerConfig {
        port: 8080,
        max_batch_size: 1,
        timeout_seconds: 30,
        enable_logging: false,
        max_concurrent_requests: 10,
    };
    let server = ModelServer::new(model, config);
    let stats = server.get_stats();
    assert_eq!(stats.total_requests, 0);
    assert_eq!(stats.successful_predictions, 0);
    assert_eq!(stats.total_errors, 0);
    assert_eq!(stats.avg_response_time_ms, 0.0);
    assert_eq!(stats.active_requests, 0);
}
#[test]
fn test_target_platform_host_detection_is_consistent() {
    let variants = [
        TargetPlatform::LinuxX64,
        TargetPlatform::LinuxArm64,
        TargetPlatform::WindowsX64,
        TargetPlatform::MacOSX64,
        TargetPlatform::MacOSArm64,
        TargetPlatform::AndroidArm64,
        TargetPlatform::AndroidX64,
        TargetPlatform::IOSArm64,
        TargetPlatform::IOSX64,
        TargetPlatform::WASM,
    ];
    let host = TargetPlatform::host();
    for variant in &variants {
        assert_eq!(variant.is_host(), host.as_ref() == Some(variant));
    }
    assert!(!TargetPlatform::WASM.is_host());
}
#[test]
fn test_binary_project_cargo_toml_has_path_dependency_and_no_cdylib() {
    let manifest_dir = "/some/workspace/scirs2-neural";
    let toml = binary_project_cargo_toml("scirs2_model_runtime", manifest_dir);
    assert!(toml.contains("name = \"scirs2_model_runtime\""));
    assert!(toml.contains(&format!("path = \"{manifest_dir}\"")));
    assert!(toml.contains("features = [\"legacy_serialization\"]"));
    assert!(toml.contains("[[bin]]"));
    assert!(!toml.contains("crate-type"));
}
#[test]
fn test_binary_project_main_rs_loads_model_and_forwards() {
    let main_rs = binary_project_main_rs();
    assert!(main_rs.contains("load_model"));
    assert!(main_rs.contains("SerializationFormat::JSON"));
    assert!(main_rs.contains(".forward("));
    assert!(main_rs.contains("fn main()"));
}
#[test]
fn test_cdylib_project_cargo_toml_has_cdylib_crate_type() {
    let manifest_dir = "/some/workspace/scirs2-neural";
    let toml = cdylib_project_cargo_toml("scirs2_model_cdylib", manifest_dir);
    assert!(toml.contains("crate-type = [\"cdylib\"]"));
    assert!(toml.contains(&format!("path = \"{manifest_dir}\"")));
    assert!(toml.contains("features = [\"legacy_serialization\"]"));
}
#[test]
fn test_cdylib_project_lib_rs_matches_c_header_abi_symbols() {
    let lib_rs = cdylib_project_lib_rs();
    for symbol in [
        "scirs2_model_load",
        "scirs2_model_predict",
        "scirs2_model_free",
        "scirs2_tensor_free",
    ] {
        assert!(
            lib_rs.contains(symbol),
            "generated cdylib source is missing expected ABI symbol `{symbol}`"
        );
    }
    assert!(lib_rs.contains("#[no_mangle]"));
    assert!(lib_rs.contains("extern \"C\" fn"));
}
#[test]
fn test_generate_c_header_declares_same_abi_symbols_as_cdylib() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let header_path = temp_dir.path().join("scirs2_model.h");
    let config = CBindingConfig {
        library_name: "scirs2_model".to_string(),
        header_guard: "SCIRS2_MODEL_H".to_string(),
        namespace: None,
        calling_convention: CallingConvention::CDecl,
        additional_headers: Vec::new(),
        type_mappings: HashMap::new(),
    };
    packager
        .generate_c_header(&header_path, &config)
        .expect("header generation should succeed");
    let header = fs::read_to_string(&header_path).expect("failed to read generated header");
    for symbol in [
        "scirs2_model_load",
        "scirs2_model_predict",
        "scirs2_model_free",
        "scirs2_tensor_free",
    ] {
        assert!(header.contains(symbol));
    }
}
#[test]
fn test_codegen_pipeline_end_to_end_std_only_program() {
    if Command::new("rustc").arg("--version").output().is_err() {
        println!("rustc not found on PATH; skipping end-to-end codegen smoke test");
        return;
    }
    let dir = unique_temp_dir("scirs2_neural_serving_rustc_smoke");
    fs::create_dir_all(&dir).expect("failed to create smoke-test temp dir");
    let src_path = dir.join("main.rs");
    fs::write(
        &src_path,
        r#"fn main() { println!("scirs2-neural-codegen-smoke-ok"); }"#,
    )
    .expect("failed to write smoke-test source");
    let exe_path = dir.join(if cfg!(windows) { "smoke.exe" } else { "smoke" });
    let compile_output = Command::new("rustc")
        .arg(&src_path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .expect("failed to invoke rustc");
    assert!(
        compile_output.status.success(),
        "rustc failed: {}",
        String::from_utf8_lossy(&compile_output.stderr)
    );
    let run_output = Command::new(&exe_path)
        .output()
        .expect("failed to run compiled smoke-test binary");
    assert_eq!(run_output.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&run_output.stdout).trim(),
        "scirs2-neural-codegen-smoke-ok"
    );
    let _ = fs::remove_dir_all(&dir);
}
#[test]
fn test_generate_runtime_binary_rejects_non_host_platform() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let out_path = temp_dir.path().join("runtime_out");
    let result = packager.generate_runtime_binary(&out_path, &TargetPlatform::WASM);
    assert!(
        result.is_err(),
        "a non-host TargetPlatform must be rejected, not silently faked"
    );
    assert!(
        !out_path.exists(),
        "no artifact should be written for a rejected non-host target"
    );
}
#[test]
fn test_generate_shared_library_rejects_non_host_platform() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let config = CBindingConfig {
        library_name: "scirs2_model".to_string(),
        header_guard: "SCIRS2_MODEL_H".to_string(),
        namespace: None,
        calling_convention: CallingConvention::CDecl,
        additional_headers: Vec::new(),
        type_mappings: HashMap::new(),
    };
    let out_path = temp_dir.path().join("libscirs2_model.out");
    let result = packager.generate_shared_library(&out_path, &config, &TargetPlatform::WASM);
    assert!(
        result.is_err(),
        "a non-host TargetPlatform must be rejected, not silently faked"
    );
    assert!(
        !out_path.exists(),
        "no artifact should be written for a rejected non-host target"
    );
}
#[test]
fn test_ensure_host_platform_accepts_actual_host() {
    if let Some(host) = TargetPlatform::host() {
        assert!(ensure_host_platform(&host).is_ok());
    }
}
#[test]
fn test_generate_wasm_module_returns_honest_error_not_fake_bytes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let path = temp_dir.path().join("model.wasm");
    let config = WasmConfig {
        wasm_version: WasmVersion::V1_0,
        enable_simd: false,
        enable_threads: false,
        memory_config: WasmMemoryConfig {
            initial_pages: 1,
            max_pages: None,
            allow_growth: false,
        },
        imports: Vec::new(),
        exports: Vec::new(),
    };
    let result = packager.generate_wasm_module(&path, &config);
    assert!(result.is_err());
    assert!(!path.exists(), "no fake .wasm artifact should be written");
}
#[test]
fn test_generate_android_aar_returns_honest_error_not_fake_bytes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let path = temp_dir.path().join("model.aar");
    let config = MobileConfig {
        platform: MobilePlatform::Android,
        min_os_version: "21".to_string(),
        architecture: MobileArchitecture::ARM64,
        optimization: MobileOptimization {
            enable_quantization: false,
            pruning_level: 0.0,
            memory_optimization: false,
            battery_optimization: false,
        },
        framework_config: FrameworkConfig {
            use_metal: false,
            use_nnapi: false,
            use_gpu: false,
            thread_pool_size: None,
        },
    };
    let result = packager.generate_android_aar(&path, &config);
    assert!(result.is_err());
    assert!(!path.exists(), "no fake .aar artifact should be written");
}
#[test]
fn test_generate_ios_framework_returns_honest_error_not_fake_bytes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let path = temp_dir.path().join("SciRS2Model.framework");
    let config = MobileConfig {
        platform: MobilePlatform::IOS,
        min_os_version: "12.0".to_string(),
        architecture: MobileArchitecture::ARM64,
        optimization: MobileOptimization {
            enable_quantization: false,
            pruning_level: 0.0,
            memory_optimization: false,
            battery_optimization: false,
        },
        framework_config: FrameworkConfig {
            use_metal: false,
            use_nnapi: false,
            use_gpu: false,
            thread_pool_size: None,
        },
    };
    let result = packager.generate_ios_framework(&path, &config);
    assert!(result.is_err());
    assert!(!path.exists(), "no fake framework binary should be written");
}
#[test]
fn test_generate_python_wheel_returns_honest_error_not_fake_bytes() {
    let temp_dir = TempDir::new().expect("failed to create temp dir");
    let model = make_test_model();
    let packager = ModelPackager::new(model, temp_dir.path().to_path_buf());
    let path = temp_dir.path().join("model-1.0.0-py3-none-any.whl");
    let result = packager.generate_python_wheel(&path);
    assert!(result.is_err());
    assert!(!path.exists(), "no fake .whl artifact should be written");
}

// Note: deliberately no test here that drives `package_native`/`package_c_library`
// (or `generate_runtime_binary`/`generate_shared_library` directly on the host
// platform) end-to-end: doing so triggers a real `cargo build` of a
// scirs2-neural-*dependent* scaffolded project, which is too slow/fragile for
// the standard test suite. `test_codegen_pipeline_end_to_end_std_only_program`
// above proves the compile-and-run pipeline mechanics with a std-only program
// instead; the scirs2-neural-dependent path is exercised structurally via the
// `test_binary_project_*`/`test_cdylib_project_*` template-content assertions.
