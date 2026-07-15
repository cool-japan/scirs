//! Shared configuration, metadata, and result types for model packaging and
//! serving, plus the (non-code-generation) [`ModelServer`] runtime.

use crate::error::Result;
use crate::models::sequential::Sequential;
use crate::models::Model;
use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::path::PathBuf;

/// Result of model packaging operation
#[derive(Debug, Clone)]
pub struct PackageResult {
    /// Package format used
    pub format: PackageFormat,
    /// Target platform
    pub platform: TargetPlatform,
    /// Generated output file paths
    pub output_paths: Vec<PathBuf>,
    /// Package metadata
    pub metadata: PackageMetadata,
}
/// Mobile deployment configuration
#[derive(Debug, Clone)]
pub struct MobileConfig {
    /// Target mobile platform
    pub platform: MobilePlatform,
    /// Minimum OS version
    pub min_os_version: String,
    /// Deployment architecture
    pub architecture: MobileArchitecture,
    /// Optimization settings
    pub optimization: MobileOptimization,
    /// Framework configuration
    pub framework_config: FrameworkConfig,
}
/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server port
    pub port: u16,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Enable request logging
    pub enable_logging: bool,
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
}
/// Mobile optimization settings
#[derive(Debug, Clone)]
pub struct MobileOptimization {
    /// Enable quantization
    pub enable_quantization: bool,
    /// Model pruning level (0.0 to 1.0)
    pub pruning_level: f64,
    /// Memory optimization
    pub memory_optimization: bool,
    /// Battery optimization
    pub battery_optimization: bool,
}
/// Target platform for deployment
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    /// Linux x86_64
    LinuxX64,
    /// Linux ARM64
    LinuxArm64,
    /// Windows x86_64
    WindowsX64,
    /// macOS x86_64
    MacOSX64,
    /// macOS ARM64 (Apple Silicon)
    MacOSArm64,
    /// Android ARM64
    AndroidArm64,
    /// Android x86_64
    AndroidX64,
    /// iOS ARM64
    IOSArm64,
    /// iOS x86_64 (Simulator)
    IOSX64,
    /// WebAssembly
    WASM,
}
impl TargetPlatform {
    /// Detect the [`TargetPlatform`] that matches the machine this code is
    /// currently executing on, if it is one of the desktop targets that
    /// [`ModelPackager`]'s host code generation pipeline knows how to build for.
    ///
    /// Returns `None` for hosts whose OS/architecture combination doesn't map onto
    /// any supported desktop variant (e.g. a 32-bit x86 build host, or a platform
    /// this module has no native build support for).
    pub fn host() -> Option<Self> {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("linux", "x86_64") => Some(Self::LinuxX64),
            ("linux", "aarch64") => Some(Self::LinuxArm64),
            ("windows", "x86_64") => Some(Self::WindowsX64),
            ("macos", "x86_64") => Some(Self::MacOSX64),
            ("macos", "aarch64") => Some(Self::MacOSArm64),
            _ => None,
        }
    }
    /// Whether `self` is the platform this code is currently executing on.
    pub fn is_host(&self) -> bool {
        TargetPlatform::host().as_ref() == Some(self)
    }
    /// Executable file suffix for this platform (e.g. `.exe` on Windows).
    pub(super) fn exe_suffix(&self) -> &'static str {
        match self {
            TargetPlatform::WindowsX64 => ".exe",
            _ => "",
        }
    }
    /// `(prefix, suffix)` for a `cdylib` artifact name on this platform,
    /// e.g. `("lib", ".so")` on Linux or `("", ".dll")` on Windows.
    pub(super) fn dylib_prefix_suffix(&self) -> (&'static str, &'static str) {
        match self {
            TargetPlatform::WindowsX64 => ("", ".dll"),
            TargetPlatform::MacOSX64 | TargetPlatform::MacOSArm64 => ("lib", ".dylib"),
            _ => ("lib", ".so"),
        }
    }
}
/// WebAssembly import specification
#[derive(Debug, Clone)]
pub struct WasmImport {
    /// Module name
    pub module: String,
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: String,
}
/// WebAssembly memory configuration
#[derive(Debug, Clone)]
pub struct WasmMemoryConfig {
    /// Initial memory pages (64KB each)
    pub initial_pages: usize,
    /// Maximum memory pages
    pub max_pages: Option<usize>,
    /// Enable memory growth
    pub allow_growth: bool,
}
/// Model package format for deployment
#[derive(Debug, Clone, PartialEq)]
pub enum PackageFormat {
    /// Standard Rust binary
    Native,
    /// WebAssembly binary
    WebAssembly,
    /// C/C++ shared library
    CSharedLibrary,
    /// Android AAR package
    AndroidAAR,
    /// iOS Framework
    IOSFramework,
    /// Python wheel
    PythonWheel,
    /// Docker container
    Docker,
}
/// GPU requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRequirements {
    /// Minimum GPU memory in MB
    pub min_memory_mb: usize,
    /// Required compute capability
    pub compute_capability: Option<String>,
    /// Required drivers
    pub drivers: Vec<String>,
}
/// Mobile architecture specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileArchitecture {
    /// ARM64 architecture
    ARM64,
    /// x86_64 architecture (simulators)
    X86_64,
    /// Universal (fat binary)
    Universal,
}
/// CPU requirements specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuRequirements {
    /// Minimum CPU cores
    pub min_cores: usize,
    /// Required instruction sets
    pub instruction_sets: Vec<String>,
    /// Minimum frequency in MHz
    pub min_frequency_mhz: Option<usize>,
}
/// Model serving runtime
pub struct ModelServer<F: Float + Debug + scirs2_core::ndarray::ScalarOperand + NumAssign + 'static>
{
    /// Loaded model
    pub(super) model: Sequential<F>,
    /// Server configuration
    pub(super) config: ServerConfig,
    /// Runtime statistics
    pub(super) stats: ServerStats,
}
impl<
        F: Float
            + Debug
            + scirs2_core::ndarray::ScalarOperand
            + FromPrimitive
            + Display
            + NumAssign
            + 'static,
    > ModelServer<F>
{
    /// Create a new model server
    pub fn new(model: Sequential<F>, config: ServerConfig) -> Self {
        Self {
            model,
            config,
            stats: ServerStats {
                total_requests: 0,
                successful_predictions: 0,
                total_errors: 0,
                avg_response_time_ms: 0.0,
                active_requests: 0,
            },
        }
    }
    /// Start the model server (stub implementation: in a real deployment this
    /// would start an HTTP server using a framework like `axum`).
    pub fn start(&mut self) -> Result<()> {
        println!("Starting SciRS2 Model Server on port {}", self.config.port);
        println!("Max batch size: {}", self.config.max_batch_size);
        println!("Timeout: {}s", self.config.timeout_seconds);
        Ok(())
    }
    /// Process prediction request
    pub fn predict(&mut self, input: &ArrayD<F>) -> Result<ArrayD<F>> {
        self.stats.total_requests += 1;
        self.stats.active_requests += 1;
        let start_time = std::time::Instant::now();
        let result = self.model.forward(input);
        let elapsed = start_time.elapsed();
        self.stats.active_requests -= 1;
        match result {
            Ok(output) => {
                self.stats.successful_predictions += 1;
                self.update_response_time(elapsed.as_millis() as f64);
                Ok(output)
            }
            Err(e) => {
                self.stats.total_errors += 1;
                Err(e)
            }
        }
    }
    /// Get server statistics
    pub fn get_stats(&self) -> &ServerStats {
        &self.stats
    }
    pub(super) fn update_response_time(&mut self, response_time_ms: f64) {
        let total_responses = self.stats.successful_predictions + self.stats.total_errors;
        if total_responses > 0 {
            self.stats.avg_response_time_ms =
                (self.stats.avg_response_time_ms * (total_responses - 1) as f64 + response_time_ms)
                    / total_responses as f64;
        }
    }
}
/// Server runtime statistics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServerStats {
    /// Total requests served
    pub total_requests: u64,
    /// Total successful predictions
    pub successful_predictions: u64,
    /// Total errors
    pub total_errors: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Current active requests
    pub active_requests: usize,
}
/// Mobile platform specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobilePlatform {
    /// iOS platform
    IOS,
    /// Android platform
    Android,
    /// Both platforms
    Universal,
}
/// Framework configuration for mobile
#[derive(Debug, Clone)]
pub struct FrameworkConfig {
    /// iOS: use Metal Performance Shaders
    pub use_metal: bool,
    /// Android: use NNAPI
    pub use_nnapi: bool,
    /// Use GPU acceleration
    pub use_gpu: bool,
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
}
/// Model package metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Model description
    pub description: String,
    /// Author information
    pub author: String,
    /// License information
    pub license: String,
    /// Target platforms
    pub platforms: Vec<String>,
    /// Dependencies
    pub dependencies: HashMap<String, String>,
    /// Model input specifications
    pub input_specs: Vec<TensorSpec>,
    /// Model output specifications
    pub output_specs: Vec<TensorSpec>,
    /// Runtime requirements
    pub runtime_requirements: RuntimeRequirements,
    /// Packaging timestamp
    pub timestamp: String,
    /// Model checksum
    pub checksum: String,
}
/// Tensor specification for inputs/outputs
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape (None for dynamic dimensions)
    pub shape: Vec<Option<usize>>,
    /// Data type
    pub dtype: String,
    /// Optional description
    pub description: Option<String>,
    /// Value range (min, max)
    pub range: Option<(f64, f64)>,
}
/// Calling convention for C bindings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallingConvention {
    /// C calling convention
    CDecl,
    /// Standard calling convention
    StdCall,
    /// Fast calling convention
    FastCall,
}
/// Runtime requirements for deployment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirements {
    /// Minimum memory requirement in MB
    pub min_memory_mb: usize,
    /// CPU requirements
    pub cpu_requirements: CpuRequirements,
    /// GPU requirements (optional)
    pub gpu_requirements: Option<GpuRequirements>,
    /// Additional system dependencies
    pub system_dependencies: Vec<String>,
}
/// WebAssembly compilation configuration
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Target WebAssembly version
    pub wasm_version: WasmVersion,
    /// Enable SIMD instructions
    pub enable_simd: bool,
    /// Enable multi-threading
    pub enable_threads: bool,
    /// Memory configuration
    pub memory_config: WasmMemoryConfig,
    /// Import/export configurations
    pub imports: Vec<WasmImport>,
    /// Export functions
    pub exports: Vec<String>,
}
/// C/C++ binding configuration
#[derive(Debug, Clone)]
pub struct CBindingConfig {
    /// Library name
    pub library_name: String,
    /// Include header guard
    pub header_guard: String,
    /// Namespace (for C++)
    pub namespace: Option<String>,
    /// Export calling convention
    pub calling_convention: CallingConvention,
    /// Include additional headers
    pub additional_headers: Vec<String>,
    /// Custom type mappings
    pub type_mappings: HashMap<String, String>,
}
/// WebAssembly version target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmVersion {
    /// WebAssembly 1.0
    V1_0,
    /// WebAssembly 2.0 (future)
    V2_0,
}
/// Optimization level for deployment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization - fastest compilation
    None,
    /// Basic optimization - balanced speed/size
    Basic,
    /// Aggressive optimization - best performance
    Aggressive,
    /// Size optimization - smallest binary
    Size,
}
