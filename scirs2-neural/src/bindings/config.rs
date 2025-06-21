//! Configuration types and structures for C/C++ binding generation
//!
//! This module defines all the configuration options and data structures
//! needed to customize the C/C++ binding generation process.

use std::collections::HashMap;
use std::path::PathBuf;

/// C/C++ binding generation configuration
#[derive(Debug, Clone)]
pub struct BindingConfig {
    /// Library name
    pub library_name: String,
    /// Target language (C or C++)
    pub language: BindingLanguage,
    /// API style configuration
    pub api_style: ApiStyle,
    /// Type mapping configuration
    pub type_mappings: TypeMappings,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Error handling approach
    pub error_handling: ErrorHandling,
    /// Threading configuration
    pub threading: ThreadingConfig,
    /// Build system configuration
    pub build_system: BuildSystemConfig,
}

/// Target binding language
#[derive(Debug, Clone, PartialEq)]
pub enum BindingLanguage {
    /// Pure C bindings
    C,
    /// C++ bindings with classes
    Cpp,
    /// C bindings with C++ wrapper
    CWithCppWrapper,
}

/// API style for bindings
#[derive(Debug, Clone, PartialEq)]
pub enum ApiStyle {
    /// Procedural API (function-based)
    Procedural,
    /// Object-oriented API (class-based)
    ObjectOriented,
    /// Hybrid approach
    Hybrid,
}

/// Type mapping configuration
#[derive(Debug, Clone)]
pub struct TypeMappings {
    /// Primitive type mappings
    pub primitives: HashMap<String, String>,
    /// Array type mappings
    pub arrays: ArrayMapping,
    /// String handling
    pub strings: StringMapping,
    /// Custom type definitions
    pub custom_types: Vec<CustomType>,
}

/// Array type mapping strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayMapping {
    /// Use plain C arrays with separate size parameter
    PlainArrays,
    /// Use structure with data pointer and metadata
    StructuredArrays,
    /// Use custom array type
    CustomArrayType(String),
}

/// String handling strategy
#[derive(Debug, Clone, PartialEq)]
pub enum StringMapping {
    /// Use null-terminated C strings
    CString,
    /// Use length-prefixed strings
    LengthPrefixed,
    /// Use custom string type
    CustomString(String),
}

/// Custom type definition
#[derive(Debug, Clone)]
pub struct CustomType {
    /// Type name in Rust
    pub rust_name: String,
    /// Type name in C/C++
    pub c_name: String,
    /// Type definition
    pub definition: String,
    /// Include dependencies
    pub includes: Vec<String>,
}

/// Memory management strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryStrategy {
    /// Manual memory management
    Manual,
    /// Reference counting
    ReferenceCounting,
    /// RAII with smart pointers (C++ only)
    RAII,
    /// Custom allocator
    CustomAllocator(String),
}

/// Error handling approach
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorHandling {
    /// Return error codes
    ErrorCodes,
    /// Use errno
    Errno,
    /// C++ exceptions (C++ only)
    Exceptions,
    /// Callback-based error handling
    Callbacks,
}

/// Threading configuration
#[derive(Debug, Clone)]
pub struct ThreadingConfig {
    /// Thread safety level
    pub safety_level: ThreadSafety,
    /// Synchronization primitives
    pub sync_primitives: Vec<SyncPrimitive>,
    /// Thread pool configuration
    pub thread_pool: Option<ThreadPoolConfig>,
}

/// Thread safety level
#[derive(Debug, Clone, PartialEq)]
pub enum ThreadSafety {
    /// Not thread-safe
    None,
    /// Thread-safe for reads
    ReadOnly,
    /// Fully thread-safe
    Full,
    /// Thread-local storage
    ThreadLocal,
}

/// Synchronization primitive
#[derive(Debug, Clone, PartialEq)]
pub enum SyncPrimitive {
    /// Mutex
    Mutex,
    /// Read-write lock
    RwLock,
    /// Atomic operations
    Atomic,
    /// Condition variables
    CondVar,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Default thread count
    pub default_threads: Option<usize>,
    /// Minimum threads
    pub min_threads: usize,
    /// Maximum threads
    pub max_threads: usize,
    /// Thread naming pattern
    pub thread_name_pattern: String,
}

/// Build system configuration
#[derive(Debug, Clone)]
pub struct BuildSystemConfig {
    /// Target build system
    pub system: BuildSystem,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
    /// Linker flags
    pub linker_flags: Vec<String>,
    /// Dependencies
    pub dependencies: Vec<Dependency>,
    /// Install configuration
    pub install_config: InstallConfig,
}

/// Build system type
#[derive(Debug, Clone, PartialEq)]
pub enum BuildSystem {
    /// CMake
    CMake,
    /// GNU Make
    Make,
    /// Meson
    Meson,
    /// Bazel
    Bazel,
    /// Custom build system
    Custom(String),
}

/// External dependency
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Dependency name
    pub name: String,
    /// Version requirement
    pub version: Option<String>,
    /// Required headers
    pub headers: Vec<String>,
    /// Required libraries
    pub libraries: Vec<String>,
    /// Package manager (pkg-config, vcpkg, etc.)
    pub package_manager: Option<String>,
}

/// Installation configuration
#[derive(Debug, Clone)]
pub struct InstallConfig {
    /// Installation prefix
    pub prefix: String,
    /// Binary directory
    pub bin_dir: String,
    /// Library directory
    pub lib_dir: String,
    /// Include directory
    pub include_dir: String,
    /// Generate pkg-config file
    pub generate_pkgconfig: bool,
}

/// Generated binding result
#[derive(Debug, Clone)]
pub struct BindingResult {
    /// Generated header files
    pub headers: Vec<PathBuf>,
    /// Generated source files
    pub sources: Vec<PathBuf>,
    /// Build system files
    pub build_files: Vec<PathBuf>,
    /// Example files
    pub examples: Vec<PathBuf>,
    /// Documentation files
    pub documentation: Vec<PathBuf>,
}

impl Default for BindingConfig {
    fn default() -> Self {
        let mut primitives = HashMap::new();
        primitives.insert("f32".to_string(), "float".to_string());
        primitives.insert("f64".to_string(), "double".to_string());
        primitives.insert("i32".to_string(), "int32_t".to_string());
        primitives.insert("u32".to_string(), "uint32_t".to_string());

        Self {
            library_name: "scirs2_model".to_string(),
            language: BindingLanguage::C,
            api_style: ApiStyle::Procedural,
            type_mappings: TypeMappings {
                primitives,
                arrays: ArrayMapping::StructuredArrays,
                strings: StringMapping::CString,
                custom_types: Vec::new(),
            },
            memory_strategy: MemoryStrategy::Manual,
            error_handling: ErrorHandling::ErrorCodes,
            threading: ThreadingConfig {
                safety_level: ThreadSafety::ReadOnly,
                sync_primitives: vec![SyncPrimitive::Mutex],
                thread_pool: None,
            },
            build_system: BuildSystemConfig {
                system: BuildSystem::CMake,
                compiler_flags: vec!["-Wall".to_string(), "-Wextra".to_string()],
                linker_flags: Vec::new(),
                dependencies: Vec::new(),
                install_config: InstallConfig {
                    prefix: "/usr/local".to_string(),
                    bin_dir: "bin".to_string(),
                    lib_dir: "lib".to_string(),
                    include_dir: "include".to_string(),
                    generate_pkgconfig: true,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_config_default() {
        let config = BindingConfig::default();
        assert_eq!(config.library_name, "scirs2_model");
        assert_eq!(config.language, BindingLanguage::C);
        assert_eq!(config.api_style, ApiStyle::Procedural);
    }

    #[test]
    fn test_type_mappings() {
        let config = BindingConfig::default();
        assert!(config.type_mappings.primitives.contains_key("f32"));
        assert_eq!(config.type_mappings.primitives["f32"], "float");
    }

    #[test]
    fn test_custom_type() {
        let custom_type = CustomType {
            rust_name: "MyStruct".to_string(),
            c_name: "my_struct_t".to_string(),
            definition: "typedef struct { int x; float y; } my_struct_t;".to_string(),
            includes: vec!["stdint.h".to_string()],
        };

        assert_eq!(custom_type.rust_name, "MyStruct");
        assert_eq!(custom_type.c_name, "my_struct_t");
    }

    #[test]
    fn test_threading_config() {
        let config = BindingConfig::default();
        assert_eq!(config.threading.safety_level, ThreadSafety::ReadOnly);
        assert!(config
            .threading
            .sync_primitives
            .contains(&SyncPrimitive::Mutex));
        assert!(config.threading.thread_pool.is_none());
    }

    #[test]
    fn test_build_system_config() {
        let config = BindingConfig::default();
        assert_eq!(config.build_system.system, BuildSystem::CMake);
        assert!(config
            .build_system
            .compiler_flags
            .contains(&"-Wall".to_string()));
        assert!(config.build_system.install_config.generate_pkgconfig);
    }
}
