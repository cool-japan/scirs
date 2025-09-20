//! API Stability Guarantees for scirs2-datasets
//!
//! This module defines the API stability levels and compatibility guarantees
//! for the scirs2-datasets crate. It provides documentation and markers for
//! different stability levels of the public API.

/// API stability levels for different components of the crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityLevel {
    /// Stable API - Guaranteed backward compatibility within major versions
    ///
    /// These APIs will not change in breaking ways within the same major version.
    /// New functionality may be added with minor version bumps, but existing
    /// functionality will remain backward compatible.
    Stable,

    /// Experimental API - Subject to change in minor versions
    ///
    /// These APIs are considered experimental and may change or be removed
    /// in future minor versions. Use with caution in production code.
    Experimental,

    /// Internal API - Not part of the public API
    ///
    /// These APIs are for internal use only and may change without notice.
    /// They should not be used by external code.
    Internal,
}

/// Trait to mark API stability level for types and functions
pub trait ApiStability {
    /// The stability level of this API component
    const STABILITY: StabilityLevel;

    /// Optional version when this API was introduced
    const INTRODUCED_IN: Option<&'static str> = None;

    /// Optional version when this API was stabilized (if applicable)
    const STABILIZED_IN: Option<&'static str> = None;

    /// Optional deprecation information
    const DEPRECATED_IN: Option<&'static str> = None;
}

/// Macro to easily annotate APIs with stability information
#[macro_export]
macro_rules! api_stability {
    (stable, $item:item) => {
        #[doc = " **API Stability: Stable** - Guaranteed backward compatibility within major versions"]
        $item
    };
    (experimental, $item:item) => {
        #[doc = " **API Stability: Experimental** - Subject to change in minor versions"]
        $item
    };
    (internal, $item:item) => {
        #[doc = " **API Stability: Internal** - Not part of the public API"]
        $item
    };
}

/// Current API version and compatibility matrix
pub struct ApiVersion;

impl ApiVersion {
    /// Current API version
    pub const CURRENT: &'static str = "0.1.0-beta.2";

    /// Minimum supported API version for backward compatibility
    pub const MIN_SUPPORTED: &'static str = "0.1.0";

    /// Next planned API version
    pub const NEXT_PLANNED: &'static str = "0.2.0";
}

/// API compatibility guarantees
///
/// This structure documents the compatibility guarantees for different
/// components of the scirs2-datasets crate.
pub struct CompatibilityGuarantees;

impl CompatibilityGuarantees {
    /// Core dataset types and basic operations
    ///
    /// These are considered stable and will maintain backward compatibility:
    /// - `Dataset` struct and its public methods
    /// - Basic dataset loading functions (`load_iris`, `load_boston`, etc.)
    /// - Core data generation functions (`make_classification`, `make_regression`, etc.)
    /// - Cross-validation utilities
    /// - Basic dataset manipulation functions
    pub const CORE_API_STABLE: &'static str = "
    The following APIs are considered stable and will maintain backward compatibility:
    
    - Dataset struct and its public methods (n_samples, n_features, etc.)
    - Toy dataset loaders (load_iris, load_boston, load_breast_cancer, etc.)
    - Core data generators (make_classification, make_regression, make_blobs, etc.)
    - Cross-validation utilities (k_fold_split, stratified_k_fold_split, etc.)
    - Basic dataset utilities (train_test_split, normalize_features, etc.)
    - Error types and Result definitions
    ";

    /// Advanced features that are experimental
    ///
    /// These features may change in minor versions:
    /// - GPU acceleration APIs
    /// - Cloud storage integration
    /// - Distributed processing
    /// - Advanced ML pipeline integration
    /// - Domain-specific datasets
    pub const EXPERIMENTAL_API: &'static str = "
    The following APIs are experimental and may change in minor versions:
    
    - GPU acceleration (gpu module)
    - Cloud storage integration (cloud module)
    - Distributed processing (distributed module)
    - Advanced ML pipeline features (ml_integration advanced features)
    - Domain-specific datasets (domain_specific module)
    - Streaming dataset processing (streaming module)
    - Advanced data generators (advanced_generators module)
    ";

    /// Internal APIs that may change without notice
    ///
    /// These are not part of the public API:
    /// - Cache implementation details
    /// - Internal data structures
    /// - Private utility functions
    pub const INTERNAL_API: &'static str = "
    The following APIs are internal and may change without notice:
    
    - Cache implementation details
    - Internal registry structures
    - Private utility functions
    - Internal error handling mechanisms
    - Performance optimization internals
    ";
}

/// Deprecation policy
///
/// This structure documents the deprecation policy for the crate.
pub struct DeprecationPolicy;

impl DeprecationPolicy {
    /// Deprecation timeline
    ///
    /// APIs marked as deprecated will be removed according to this timeline:
    /// - Major version (x.0.0): Deprecated APIs may be removed
    /// - Minor version (0.x.0): Stable APIs will not be deprecated
    /// - Patch version (0.0.x): No API changes except bug fixes
    pub const TIMELINE: &'static str = "
    Deprecation Timeline:
    
    - Major version bumps (x.0.0): Deprecated APIs may be removed
    - Minor version bumps (0.x.0): Stable APIs will not be deprecated, 
      experimental APIs may be deprecated
    - Patch version bumps (0.0.x): No API changes except bug fixes
    
    Deprecation Process:
    1. API is marked as deprecated with #[deprecated] attribute
    2. Alternative API is provided (if applicable)
    3. Deprecation notice is included in release notes
    4. API remains available for at least one major version
    5. API is removed in subsequent major version
    ";

    /// Migration guidelines
    pub const MIGRATION: &'static str = "
    Migration Guidelines:
    
    When APIs are deprecated, migration paths will be provided:
    - Clear documentation of replacement APIs
    - Migration examples in release notes
    - Automated migration tools when possible
    - Community support during transition periods
    ";
}

/// Version compatibility matrix
///
/// This documents which versions are compatible with each other.
pub struct CompatibilityMatrix;

impl CompatibilityMatrix {
    /// Check if two API versions are compatible
    pub fn is_compatible(current: &str, required: &str) -> bool {
        // Simple semantic version compatibility check
        // In a real implementation, this would use a proper semver library
        let current_parts: Vec<&str> = current.split('.').collect();
        let required_parts: Vec<&str> = required.split('.').collect();

        if current_parts.len() < 2 || required_parts.len() < 2 {
            return false;
        }

        // Major version must match for compatibility
        current_parts[0] == required_parts[0]
    }

    /// Get the compatibility level between two versions
    pub fn compatibility_level(current: &str, required: &str) -> CompatibilityLevel {
        if Self::is_compatible(current, required) {
            CompatibilityLevel::Compatible
        } else {
            CompatibilityLevel::Incompatible
        }
    }
}

/// Compatibility levels between API versions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityLevel {
    /// Fully compatible - no changes needed
    Compatible,
    /// Incompatible - breaking changes present
    Incompatible,
}

/// Runtime API stability checking
///
/// This provides runtime checks for API stability and compatibility.
pub struct StabilityChecker;

impl StabilityChecker {
    /// Check if the current crate version supports a required API version
    pub fn supports_api_version(required_version: &str) -> bool {
        CompatibilityMatrix::is_compatible(ApiVersion::CURRENT, required_version)
    }

    /// Get the stability level of a specific API component
    pub fn get_stability_level(api_name: &str) -> StabilityLevel {
        match api_name {
            // Core stable APIs
            "Dataset"
            | "load_iris"
            | "load_boston"
            | "load_breast_cancer"
            | "load_wine"
            | "load_digits"
            | "make_classification"
            | "make_regression"
            | "make_blobs"
            | "make_circles"
            | "make_moons"
            | "k_fold_split"
            | "stratified_k_fold_split"
            | "train_test_split" => StabilityLevel::Stable,

            // Experimental APIs
            "GpuContext"
            | "CloudClient"
            | "DistributedProcessor"
            | "MLPipeline"
            | "StreamingIterator" => StabilityLevel::Experimental,

            // Everything else is considered experimental for now
            _ => StabilityLevel::Experimental,
        }
    }

    /// Validate that experimental APIs are being used appropriately
    pub fn validate_experimental_usage(api_name: &str) -> Result<(), String> {
        match Self::get_stability_level(api_name) {
            StabilityLevel::Experimental => {
                eprintln!(
                    "Warning: '{api_name}' is an experimental API and may change in future versions"
                );
                Ok(())
            }
            StabilityLevel::Internal => Err(format!(
                "Error: '{api_name}' is an internal API and should not be used directly"
            )),
            StabilityLevel::Stable => Ok(()),
        }
    }
}

/// Feature flags and their stability levels
pub mod feature_flags {
    use super::StabilityLevel;

    /// GPU acceleration features
    pub const GPU: (&str, StabilityLevel) = ("gpu", StabilityLevel::Experimental);

    /// Cloud storage features  
    pub const CLOUD: (&str, StabilityLevel) = ("cloud", StabilityLevel::Experimental);

    /// Distributed processing features
    pub const DISTRIBUTED: (&str, StabilityLevel) = ("distributed", StabilityLevel::Experimental);

    /// Advanced ML integration features
    pub const ML_ADVANCED: (&str, StabilityLevel) = ("ml_advanced", StabilityLevel::Experimental);

    /// Streaming processing features
    pub const STREAMING: (&str, StabilityLevel) = ("streaming", StabilityLevel::Experimental);

    /// Domain-specific datasets
    pub const DOMAIN_SPECIFIC: (&str, StabilityLevel) =
        ("domain_specific", StabilityLevel::Experimental);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatibility_matrix() {
        assert!(CompatibilityMatrix::is_compatible("0.1.0", "0.1.0"));
        assert!(CompatibilityMatrix::is_compatible("0.1.1", "0.1.0"));
        assert!(!CompatibilityMatrix::is_compatible("1.0.0", "0.1.0"));
        assert!(!CompatibilityMatrix::is_compatible("0.1.0", "1.0.0"));
    }

    #[test]
    fn test_stability_checker() {
        assert_eq!(
            StabilityChecker::get_stability_level("Dataset"),
            StabilityLevel::Stable
        );
        assert_eq!(
            StabilityChecker::get_stability_level("GpuContext"),
            StabilityLevel::Experimental
        );
    }

    #[test]
    fn test_api_version() {
        // Test that version strings are properly formatted (non-empty and contain dots)
        assert!(ApiVersion::CURRENT.contains('.'));
        assert!(ApiVersion::MIN_SUPPORTED.contains('.'));
        assert!(ApiVersion::NEXT_PLANNED.contains('.'));

        // Test that versions are properly ordered (basic check)
        assert!(ApiVersion::CURRENT >= ApiVersion::MIN_SUPPORTED);
    }
}
