//! Cross-platform validation utilities for consistent behavior across operating systems and architectures.
//!
//! This module provides validation utilities that handle platform-specific differences
//! in numeric formats, file systems, memory models, and hardware capabilities to ensure
//! consistent behavior across Windows, macOS, Linux, and different CPU architectures.

use crate::error::{CoreError, CoreResult, ErrorContext};
use crate::validation::production::{
    ValidationContext, ValidationError, ValidationResult, ValidationSeverity,
};
use std::collections::HashMap;

/// Platform information detected at runtime
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlatformInfo {
    /// Operating system family
    pub os_family: OsFamily,
    /// CPU architecture
    pub arch: CpuArchitecture,
    /// Available SIMD instruction sets
    pub simd_support: SimdSupport,
    /// Endianness of the target platform
    pub endianness: Endianness,
    /// Native path separator
    pub path_separator: char,
    /// Maximum file path length
    pub max_path_length: usize,
    /// Default memory page size
    pub page_size: usize,
    /// Whether the platform supports memory-mapped files
    pub memory_mapping_support: bool,
    /// Default floating-point precision behavior
    pub fp_behavior: FloatingPointBehavior,
}

/// Operating system families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsFamily {
    Windows,
    Unix, // Linux, macOS, BSD, etc.
    Wasm, // WebAssembly runtime
    Unknown,
}

/// CPU architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArchitecture {
    X86_64,
    AArch64, // ARM64
    X86,     // 32-bit x86
    ARM,     // 32-bit ARM
    RISCV64,
    PowerPC64,
    Wasm32, // WebAssembly 32-bit
    Wasm64, // WebAssembly 64-bit
    Other(u32),
}

/// SIMD instruction set support
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimdSupport {
    /// SSE support levels (x86/x64)
    pub sse: Option<SseLevel>,
    /// AVX support levels (x86/x64)
    pub avx: Option<AvxLevel>,
    /// NEON support (ARM)
    pub neon: bool,
    /// SVE support (ARM)
    pub sve: bool,
    /// Vector extension support level
    pub vector_width: usize,
}

/// SSE instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SseLevel {
    Sse,
    Sse2,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
}

/// AVX instruction set levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AvxLevel {
    Avx,
    Avx2,
    Avx512f,
    Avx512bw,
    Avx512dq,
}

/// Platform endianness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

/// Floating-point behavior characteristics
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatingPointBehavior {
    /// Whether denormal numbers are supported
    pub denormals_supported: bool,
    /// Default rounding mode
    pub rounding_mode: RoundingMode,
    /// Whether NaN propagation is IEEE 754 compliant
    pub nan_propagation: bool,
    /// Whether infinity is supported
    pub infinity_supported: bool,
}

/// Floating-point rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    ToNearest,
    TowardZero,
    TowardPositiveInfinity,
    TowardNegativeInfinity,
}

/// Cross-platform validator with platform-aware validation rules
pub struct CrossPlatformValidator {
    /// Current platform information
    platform_info: PlatformInfo,
    /// Validation context
    #[allow(dead_code)]
    context: ValidationContext,
    /// Cached validation results for performance
    cache: HashMap<String, ValidationResult>,
}

impl CrossPlatformValidator {
    /// Create a new cross-platform validator
    pub fn new() -> CoreResult<Self> {
        let platform_info = Self::detect_platform_info()?;
        Ok(Self {
            platform_info,
            context: ValidationContext::default(),
            cache: HashMap::new(),
        })
    }

    /// Create a validator with custom context
    pub fn with_context(context: ValidationContext) -> CoreResult<Self> {
        let platform_info = Self::detect_platform_info()?;
        Ok(Self {
            platform_info,
            context,
            cache: HashMap::new(),
        })
    }

    /// Detect platform information at runtime
    fn detect_platform_info() -> CoreResult<PlatformInfo> {
        let os_family = if cfg!(target_family = "wasm") {
            OsFamily::Wasm
        } else if cfg!(windows) {
            OsFamily::Windows
        } else if cfg!(unix) {
            OsFamily::Unix
        } else {
            OsFamily::Unknown
        };

        let arch = if cfg!(target_arch = "wasm32") {
            CpuArchitecture::Wasm32
        } else if cfg!(target_arch = "wasm64") {
            CpuArchitecture::Wasm64
        } else if cfg!(target_arch = "x86_64") {
            CpuArchitecture::X86_64
        } else if cfg!(target_arch = "aarch64") {
            CpuArchitecture::AArch64
        } else if cfg!(target_arch = "x86") {
            CpuArchitecture::X86
        } else if cfg!(target_arch = "arm") {
            CpuArchitecture::ARM
        } else if cfg!(target_arch = "riscv64") {
            CpuArchitecture::RISCV64
        } else if cfg!(target_arch = "powerpc64") {
            CpuArchitecture::PowerPC64
        } else {
            CpuArchitecture::Other(0)
        };

        let endianness = if cfg!(target_endian = "little") {
            Endianness::Little
        } else {
            Endianness::Big
        };

        let path_separator = if cfg!(windows) {
            '\\'
        } else {
            '/' // Unix-style paths for all non-Windows platforms (including WASM)
        };

        let max_path_length = if cfg!(target_family = "wasm") {
            1024 // Conservative limit for WASM environments
        } else if cfg!(windows) {
            260 // MAX_PATH on Windows (unless long path support is enabled)
        } else {
            4096 // Common limit on Unix systems
        };

        // Detect SIMD support
        let simd_support = Self::detect_simd_support(arch);

        // Detect system page size
        let page_size = Self::detect_page_size();

        let memory_mapping_support = !cfg!(target_family = "wasm");

        let fp_behavior = FloatingPointBehavior {
            denormals_supported: true, // Most modern platforms support denormals
            rounding_mode: RoundingMode::ToNearest,
            nan_propagation: true,
            infinity_supported: true,
        };

        Ok(PlatformInfo {
            os_family,
            arch,
            simd_support,
            endianness,
            path_separator,
            max_path_length,
            page_size,
            memory_mapping_support,
            fp_behavior,
        })
    }

    /// Detect SIMD instruction set support
    fn detect_simd_support(arch: CpuArchitecture) -> SimdSupport {
        match arch {
            CpuArchitecture::X86_64 | CpuArchitecture::X86 => {
                // For x86/x64, we'd normally use cpuid to detect features
                // For now, provide conservative defaults
                SimdSupport {
                    sse: Some(SseLevel::Sse2), // SSE2 is guaranteed on x64
                    avx: if cfg!(target_feature = "avx2") {
                        Some(AvxLevel::Avx2)
                    } else if cfg!(target_feature = "avx") {
                        Some(AvxLevel::Avx)
                    } else {
                        None
                    },
                    neon: false,
                    sve: false,
                    vector_width: if cfg!(target_feature = "avx512f") {
                        512
                    } else if cfg!(target_feature = "avx2") {
                        256
                    } else {
                        128
                    },
                }
            }
            CpuArchitecture::AArch64 | CpuArchitecture::ARM => {
                SimdSupport {
                    sse: None,
                    avx: None,
                    neon: true,        // NEON is standard on ARM64
                    sve: false,        // SVE detection would require runtime checks
                    vector_width: 128, // Default ARM NEON width
                }
            }
            CpuArchitecture::Wasm32 | CpuArchitecture::Wasm64 => {
                SimdSupport {
                    sse: None,
                    avx: None,
                    neon: false,
                    sve: false,
                    vector_width: if cfg!(target_feature = "simd128") {
                        128 // WASM SIMD128 support
                    } else {
                        64 // No SIMD support
                    },
                }
            }
            _ => {
                SimdSupport {
                    sse: None,
                    avx: None,
                    neon: false,
                    sve: false,
                    vector_width: 64, // Conservative default
                }
            }
        }
    }

    /// Detect system page size
    fn detect_page_size() -> usize {
        #[cfg(unix)]
        {
            // Most Unix systems use 4KB pages, with some using 64KB (especially ARM64)
            // For simplicity, we'll use 4KB as default since it's most common
            4096
        }
        #[cfg(windows)]
        {
            // Windows typically uses 4KB pages, but can be 64KB on some systems
            // For simplicity, use the common default
            4096
        }
        #[cfg(not(any(unix, windows)))]
        {
            4096 // Default page size
        }
    }

    /// Validate a file path for the current platform
    pub fn validate_file_path(&mut self, path: &str) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: crate::validation::production::ValidationMetrics::default(),
        };

        // Check path length limits
        if path.len() > self.platform_info.max_path_length {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "PATH_TOO_LONG".to_string(),
                message: format!(
                    "Path length {} exceeds platform maximum of {}",
                    path.len(),
                    self.platform_info.max_path_length
                ),
                field: Some(path.to_string()),
                suggestion: Some("Use shorter path or enable long path support ".to_string()),
                severity: ValidationSeverity::Error,
            });
        }

        // Platform-specific path validation
        match self.platform_info.os_family {
            OsFamily::Windows => self.validate_windows_path(path, &mut result),
            OsFamily::Unix => self.validate_unix_path(path, &mut result),
            OsFamily::Wasm => self.validate_wasm_path(path, &mut result),
            OsFamily::Unknown => {
                result
                    .warnings
                    .push("Unknown platform - basic validation only ".to_string());
            }
        }

        // Check for null bytes (invalid on all platforms)
        if path.contains('\0') {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "NULL_BYTE_IN_PATH".to_string(),
                message: "Path contains null byte ".to_string(),
                field: Some(path.to_string()),
                suggestion: Some("Remove null bytes from path ".to_string()),
                severity: ValidationSeverity::Critical,
            });
        }

        result
    }

    /// Validate Windows-specific path constraints
    fn validate_windows_path(&self, path: &str, result: &mut ValidationResult) {
        // Check for invalid characters
        let invalid_chars = r#"<>:"|?*"#.chars().collect::<Vec<_>>();
        for &ch in &invalid_chars {
            if path.contains(ch) {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "INVALID_WINDOWS_CHAR".to_string(),
                    message: format!("Character '{ch}' is invalid in Windows paths"),
                    field: Some(path.to_string()),
                    suggestion: Some("Remove or replace invalid characters".to_string()),
                    severity: ValidationSeverity::Error,
                });
                break;
            }
        }

        // Check for reserved names
        let reserved_names = [
            "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7",
            "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        ];

        let path_upper = path.to_uppercase();
        for &reserved in &reserved_names {
            if path_upper == reserved || path_upper.starts_with(&format!("{reserved}.")) {
                result.is_valid = false;
                result.errors.push(ValidationError {
                    code: "RESERVED_WINDOWS_NAME".to_string(),
                    message: format!("'{reserved}' is a reserved name on Windows"),
                    field: Some(path.to_string()),
                    suggestion: Some("Use a different filename".to_string()),
                    severity: ValidationSeverity::Error,
                });
                break;
            }
        }

        // Check for trailing spaces or periods
        if path.ends_with(' ') || path.ends_with('.') {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "INVALID_WINDOWS_ENDING".to_string(),
                message: "Windows paths cannot end with spaces or periods".to_string(),
                field: Some(path.to_string()),
                suggestion: Some("Remove trailing spaces or periods".to_string()),
                severity: ValidationSeverity::Error,
            });
        }
    }

    /// Validate Unix-specific path constraints
    fn validate_unix_path(&self, path: &str, result: &mut ValidationResult) {
        // Unix paths are generally more permissive, but check for some edge cases

        // Check for double slashes (while technically valid, often unintended)
        if path.contains("//") {
            result
                .warnings
                .push("Path contains double slashes".to_string());
        }

        // Check if path starts with /dev/, /proc/, or /sys/ - potentially dangerous
        let system_prefixes = ["/dev/", "/proc/", "/sys/"];
        for &prefix in &system_prefixes {
            if path.starts_with(prefix) {
                result.warnings.push(format!(
                    "Path accesses system directory '{prefix}' - ensure this is intended"
                ));
                break;
            }
        }

        // Check for very long path components (while Unix supports long names,
        // some filesystems have limits)
        for component in path.split('/') {
            if component.len() > 255 {
                result
                    .warnings
                    .push("Path component exceeds 255 characters".to_string());
                break;
            }
        }
    }

    /// Validate WebAssembly-specific path constraints
    fn validate_wasm_path(&self, path: &str, result: &mut ValidationResult) {
        // WebAssembly has very limited file system access

        // Check if path is attempting to access outside the sandbox
        if path.starts_with("../") || path.contains("/../") {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "WASM_SANDBOX_VIOLATION".to_string(),
                message: "WebAssembly paths cannot escape sandbox with '..'".to_string(),
                field: Some(path.to_string()),
                suggestion: Some("Use paths relative to the WASM module".to_string()),
                severity: ValidationSeverity::Critical,
            });
        }

        // Check for absolute paths (typically not allowed in WASM)
        if path.starts_with('/') {
            result.warnings.push(
                "Absolute paths may not be accessible in WebAssembly environment".to_string(),
            );
        }

        // Check for special protocols that might not work in WASM
        let special_prefixes = ["file://", "http://", "https://", "ftp://"];
        for &prefix in &special_prefixes {
            if path.starts_with(prefix) {
                result.warnings.push(format!(
                    "Protocol '{prefix}' may not be accessible in WebAssembly environment"
                ));
                break;
            }
        }

        // WASM has stricter limits on path components
        for component in path.split('/') {
            if component.len() > 128 {
                result.warnings.push(
                    "Very long path components may not be supported in WebAssembly".to_string(),
                );
                break;
            }
        }

        // Check for WASM-specific virtual file system conventions
        if path.starts_with("/tmp/") || path.starts_with("/temp/") {
            result.warnings.push(
                "Temporary directories may have limited persistence in WebAssembly".to_string(),
            );
        }

        // General warning about WASM file system limitations
        result
            .warnings
            .push("WebAssembly environment has limited file system access".to_string());
    }

    /// Validate numeric value considering platform-specific floating-point behavior
    pub fn validate_numeric_cross_platform<T>(
        &mut self,
        value: T,
        fieldname: &str,
    ) -> ValidationResult
    where
        T: PartialOrd + Copy + std::fmt::Debug + std::fmt::Display + 'static,
    {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: crate::validation::production::ValidationMetrics::default(),
        };

        // Check for platform-specific numeric issues
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
            || std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
        {
            self.validate_floating_point_value(&value, fieldname, &mut result);
        }

        // Check for endianness-sensitive operations
        if self.platform_info.endianness == Endianness::Big {
            result.warnings.push(
                "Running on big-endian platform - verify binary data compatibility".to_string(),
            );
        }

        result
    }

    /// Validate floating-point value considering platform behavior
    fn validate_floating_point_value<T>(
        &self,
        value: &T,
        fieldname: &str,
        result: &mut ValidationResult,
    ) where
        T: std::fmt::Debug + std::fmt::Display,
    {
        // This is a simplified check - in practice we'd need unsafe transmutation
        // to properly inspect the floating-point representation
        let value_str = format!("{value:?}");

        if value_str.contains("inf") && !self.platform_info.fp_behavior.infinity_supported {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "INFINITY_NOT_SUPPORTED".to_string(),
                message: format!("Infinity values not supported on this platform for {fieldname}"),
                field: Some(fieldname.to_string()),
                suggestion: Some("Use finite values only".to_string()),
                severity: ValidationSeverity::Error,
            });
        }

        if value_str.contains("nan") && !self.platform_info.fp_behavior.nan_propagation {
            result.warnings.push(format!(
                "NaN value in {fieldname} - platform may not handle NaN propagation correctly"
            ));
        }
    }

    /// Validate SIMD operation compatibility
    pub fn validate_simd_operation(
        &mut self,
        operation: &str,
        _data_size: usize,
        vector_size: usize,
    ) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: crate::validation::production::ValidationMetrics::default(),
        };

        // Check if requested vector size is supported
        if vector_size > self.platform_info.simd_support.vector_width {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "SIMD_VECTOR_TOO_LARGE".to_string(),
                message: format!(
                    "Requested vector size {} exceeds platform maximum of {}",
                    vector_size, self.platform_info.simd_support.vector_width
                ),
                field: Some(vector_size.to_string()),
                suggestion: Some(format!(
                    "Use vector size <= {}",
                    self.platform_info.simd_support.vector_width
                )),
                severity: ValidationSeverity::Error,
            });
        }

        // Check operation-specific requirements
        if operation.contains("avx") && self.platform_info.simd_support.avx.is_none() {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "AVX_NOT_SUPPORTED".to_string(),
                message: "AVX instructions not supported on this platform".to_string(),
                field: Some(operation.to_string()),
                suggestion: Some("Use SSE fallback or check platform capabilities".to_string()),
                severity: ValidationSeverity::Error,
            });
        }

        if operation.contains("neon") && !self.platform_info.simd_support.neon {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "NEON_NOT_SUPPORTED".to_string(),
                message: "NEON instructions not supported on this platform".to_string(),
                field: Some(operation.to_string()),
                suggestion: Some("Use scalar fallback".to_string()),
                severity: ValidationSeverity::Error,
            });
        }

        result
    }

    /// Validate memory allocation size considering platform limits
    pub fn validate_memory_allocation(&mut self, size: usize, purpose: &str) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metrics: crate::validation::production::ValidationMetrics::default(),
        };

        // Check if allocation is aligned to page size for optimal performance
        if size > self.platform_info.page_size && size % self.platform_info.page_size != 0 {
            result.warnings.push(format!(
                "Allocation size {} is not page-aligned (page size: {})",
                size, self.platform_info.page_size
            ));
        }

        // Platform-specific memory limits
        let max_alloc_size = match self.platform_info.arch {
            CpuArchitecture::X86 => 2usize.pow(31), // 2GB limit for 32-bit
            CpuArchitecture::ARM => 2usize.pow(31),
            CpuArchitecture::Wasm32 => 2usize.pow(31), // WASM32 has 32-bit address space
            CpuArchitecture::Wasm64 => {
                // WASM64 is limited by browser memory constraints
                4usize.pow(30) // 1GB conservative limit for WASM64
            }
            _ => usize::MAX, // 64-bit platforms
        };

        if size > max_alloc_size {
            result.is_valid = false;
            result.errors.push(ValidationError {
                code: "ALLOCATION_TOO_LARGE".to_string(),
                message: format!(
                    "Allocation size {size} exceeds platform maximum of {max_alloc_size} for {purpose}"
                ),
                field: Some(size.to_string()),
                suggestion: Some("Reduce allocation size or use memory mapping".to_string()),
                severity: ValidationSeverity::Error,
            });
        }

        // Check if memory mapping is needed but not supported
        if size > 100_000_000 && !self.platform_info.memory_mapping_support {
            result.warnings.push(format!(
                "Large allocation ({size} bytes) for {purpose} but memory mapping not supported"
            ));
        }

        result
    }

    /// Get current platform information
    pub fn platform_info(&self) -> &PlatformInfo {
        &self.platform_info
    }

    /// Clear validation cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Check if a specific platform feature is available
    pub fn is_feature_available(&self, feature: PlatformFeature) -> bool {
        match feature {
            PlatformFeature::MemoryMapping => self.platform_info.memory_mapping_support,
            PlatformFeature::Avx => self.platform_info.simd_support.avx.is_some(),
            PlatformFeature::Neon => self.platform_info.simd_support.neon,
            PlatformFeature::LongPaths => {
                // This would require more sophisticated detection in practice
                matches!(self.platform_info.os_family, OsFamily::Unix)
            }
            PlatformFeature::DenormalNumbers => self.platform_info.fp_behavior.denormals_supported,
            PlatformFeature::WasmSimd128 => {
                matches!(
                    self.platform_info.arch,
                    CpuArchitecture::Wasm32 | CpuArchitecture::Wasm64
                ) && self.platform_info.simd_support.vector_width >= 128
            }
            PlatformFeature::ThreadSupport => {
                // WASM traditionally doesn't support threads, but some environments do
                !matches!(self.platform_info.os_family, OsFamily::Wasm)
            }
            PlatformFeature::FileSystemAccess => {
                // WASM has very limited file system access
                !matches!(self.platform_info.os_family, OsFamily::Wasm)
            }
        }
    }
}

/// Platform features that can be queried
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformFeature {
    MemoryMapping,
    Avx,
    Neon,
    LongPaths,
    DenormalNumbers,
    WasmSimd128,
    ThreadSupport,
    FileSystemAccess,
}

impl Default for CrossPlatformValidator {
    fn default() -> Self {
        Self::new().expect("Failed to create cross-platform validator")
    }
}

/// Convenience functions for common cross-platform validations
/// Validate that a path is appropriate for the current platform
#[allow(dead_code)]
pub fn validate_path(path: &str) -> CoreResult<()> {
    let mut validator = CrossPlatformValidator::new()?;
    let result = validator.validate_file_path(path);

    if result.is_valid {
        Ok(())
    } else {
        Err(CoreError::ValidationError(ErrorContext::new(format!(
            "Path validation failed: {:?}",
            result.errors
        ))))
    }
}

/// Validate SIMD capability for an operation
#[allow(dead_code)]
pub fn validate_simd_capability(operation: &str, size: usize) -> CoreResult<()> {
    let mut validator = CrossPlatformValidator::new()?;
    let result = validator.validate_simd_operation(operation, size, 128);

    if result.is_valid {
        Ok(())
    } else {
        Err(CoreError::ValidationError(ErrorContext::new(format!(
            "SIMD validation failed: {:?}",
            result.errors
        ))))
    }
}

/// Get platform information
#[allow(dead_code)]
pub fn get_platform_info() -> CoreResult<PlatformInfo> {
    CrossPlatformValidator::detect_platform_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let info = CrossPlatformValidator::detect_platform_info().unwrap();

        // Basic sanity checks
        assert_ne!(info.os_family, OsFamily::Unknown);
        assert!(info.page_size > 0);
        assert!(info.max_path_length > 0);
        assert!(info.simd_support.vector_width > 0);
    }

    #[test]
    fn test_path_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Valid path
        let result = validator.validate_file_path("/home/user/data.txt");
        assert!(result.is_valid);

        // Path with null byte
        let result = validator.validate_file_path("/home/user\0/data.txt");
        assert!(!result.is_valid);
    }

    #[cfg(windows)]
    #[test]
    fn test_windows_path_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Valid Windows path
        let result = validator.validate_file_path("C:\\Users\\user\\data.txt");
        assert!(result.is_valid);

        // Invalid character
        let result = validator.validate_file_path("C:\\Users\\user<data.txt");
        assert!(!result.is_valid);

        // Reserved name
        let result = validator.validate_file_path("CON");
        assert!(!result.is_valid);
    }

    #[cfg(unix)]
    #[test]
    fn test_unix_path_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Valid Unix path
        let result = validator.validate_file_path("/home/user/data.txt");
        assert!(result.is_valid);

        // System directory warning
        let result = validator.validate_file_path("/dev/null");
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_simd_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Valid vector size
        let result = validator.validate_simd_operation("add", 128, 128);
        assert!(result.is_valid);

        // Too large vector size
        let result = validator.validate_simd_operation("add", 10000, 10000);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_memory_allocation_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Normal allocation
        let result = validator.validate_memory_allocation(1024, "test");
        assert!(result.is_valid);

        // Very large allocation
        let result = validator.validate_memory_allocation(usize::MAX - 1, "test");
        // Result depends on platform - 32-bit will fail, 64-bit might succeed
    }

    #[test]
    fn test_feature_availability() {
        let validator = CrossPlatformValidator::new().unwrap();

        // These should return boolean values without panicking
        let memory_mapping = validator.is_feature_available(PlatformFeature::MemoryMapping);
        let avx = validator.is_feature_available(PlatformFeature::Avx);
        let neon = validator.is_feature_available(PlatformFeature::Neon);
    }

    #[test]
    fn test_convenience_functions() {
        // These should not panic
        let _ = validate_path("/tmp/test.txt");
        let _ = validate_simd_capability("add", 128);
        let _ = get_platform_info();
    }

    #[test]
    fn test_wasm_specific_features() {
        let validator = CrossPlatformValidator::new().unwrap();

        // Test WASM-specific feature detection
        let wasm_simd = validator.is_feature_available(PlatformFeature::WasmSimd128);
        let thread_support = validator.is_feature_available(PlatformFeature::ThreadSupport);
        let fs_access = validator.is_feature_available(PlatformFeature::FileSystemAccess);

        // These should return boolean values without panicking
        // Test passes if we reach here without panicking
    }

    #[test]
    fn test_wasm_path_validation() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Simulate WASM environment for testing
        // Note: This test will behave differently on actual WASM vs native platforms

        // Test relative path (should be okay in WASM)
        let result = validator.validate_file_path("data/input.txt");
        // Should be valid but may have warnings in WASM

        // Test sandbox escape attempt
        let result = validator.validate_file_path("../../../etc/passwd");
        // This would be rejected in actual WASM validation

        // Just ensure these don't panic
        // Test passes if we reach here without panicking
    }

    #[test]
    fn test_platform_memory_limits() {
        let validator = CrossPlatformValidator::new().unwrap();

        // Test that memory allocation validation considers platform architecture
        let small_alloc = validator.platform_info().page_size * 2;
        let large_alloc = 2usize.pow(30); // 1GB

        // These should not panic
        let mut validator_mut = CrossPlatformValidator::new().unwrap();
        let small_result = validator_mut.validate_memory_allocation(small_alloc, "test");
        let large_result = validator_mut.validate_memory_allocation(large_alloc, "test");

        // Test passes if we reach here without panicking
    }

    #[test]
    fn test_simd_capabilities_cross_platform() {
        let mut validator = CrossPlatformValidator::new().unwrap();

        // Test SIMD validation across different architectures
        let result = validator.validate_simd_operation("generic_add", 64, 64);
        assert!(result.is_valid); // Should be supported on all platforms

        let result = validator.validate_simd_operation("avx2_multiply", 256, 256);
        // Result depends on platform - should not panic

        let result = validator.validate_simd_operation("neon_add", 128, 128);
        // Result depends on platform - should not panic

        // Test passes if we reach here without panicking
    }
}
