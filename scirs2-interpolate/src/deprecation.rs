//! Deprecation policy and warnings for SciRS2 interpolation library
//!
//! This module provides a standardized approach to handling deprecation of experimental
//! features, API changes, and maintenance of backward compatibility.
//!
//! # Overview
//!
//! The deprecation system supports:
//! - Structured deprecation warnings with migration paths
//! - Feature lifecycle management (experimental → stable → deprecated → removed)
//! - Version-based deprecation tracking
//! - Alternative function suggestions
//!
//! # Usage
//!
//! ```rust
//! use scirs2_interpolate::{deprecated_function, deprecation::DeprecationLevel};
//!
//! pub fn old_rbf_function() {
//!     deprecated_function!(
//!         name = "old_rbf_function",
//!         since = "0.1.0",
//!         reason = "Use make_enhanced_rbf_interpolator instead",
//!         alternative = "enhanced_rbf::make_enhanced_rbf_interpolator"
//!     );
//!     // Implementation
//! }
//! ```

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Global deprecation policy configuration
static DEPRECATION_CONFIG: OnceLock<Mutex<DeprecationConfig>> = OnceLock::new();

/// Deprecation configuration and tracking
#[derive(Debug, Clone)]
pub struct DeprecationConfig {
    /// Whether to show deprecation warnings
    pub show_warnings: bool,

    /// Whether to treat deprecation warnings as errors
    pub warnings_as_errors: bool,

    /// Maximum number of times to show each deprecation warning
    pub max_warning_count: usize,

    /// Track warning counts per function
    pub warning_counts: HashMap<String, usize>,

    /// Current library version for deprecation context
    pub current_version: String,
}

impl Default for DeprecationConfig {
    fn default() -> Self {
        Self {
            show_warnings: true,
            warnings_as_errors: false,
            max_warning_count: 3,
            warning_counts: HashMap::new(),
            current_version: "0.1.0-beta.2".to_string(),
        }
    }
}

/// Severity level of deprecation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeprecationLevel {
    /// Feature is experimental and may change
    Experimental,
    /// Feature is deprecated but still supported
    Soft,
    /// Feature will be removed in next major version
    Hard,
    /// Feature is removed and will cause compilation errors
    Removed,
}

/// Information about a deprecated feature
#[derive(Debug, Clone)]
pub struct DeprecationInfo {
    /// Function or feature name
    pub name: String,

    /// Version when deprecation was introduced
    pub since: String,

    /// Version when feature will be removed (if known)
    pub remove_in: Option<String>,

    /// Deprecation level
    pub level: DeprecationLevel,

    /// Reason for deprecation
    pub reason: String,

    /// Suggested alternative or migration path
    pub alternative: Option<String>,

    /// Additional notes for migration
    pub migration_notes: Option<String>,
}

/// Initialize deprecation system with default configuration
#[allow(dead_code)]
pub fn init_deprecation_system() {
    let _ = DEPRECATION_CONFIG.set(Mutex::new(DeprecationConfig::default()));
}

/// Configure deprecation warning behavior
#[allow(dead_code)]
pub fn configure_deprecation(config: DeprecationConfig) {
    init_deprecation_system();
    if let Ok(mut global_config) = DEPRECATION_CONFIG.get().unwrap().lock() {
        *global_config = config;
    }
}

/// Issue a deprecation warning if enabled
#[allow(dead_code)]
pub fn issue_deprecation_warning(info: &DeprecationInfo) {
    init_deprecation_system();

    if let Ok(mut config) = DEPRECATION_CONFIG.get().unwrap().lock() {
        if !config.show_warnings {
            return;
        }

        let count = config.warning_counts.entry(info.name.clone()).or_insert(0);
        *count += 1;

        if *count > config.max_warning_count {
            return;
        }

        let warning_msg = format_deprecation_warning(info);

        if config.warnings_as_errors {
            panic!("Deprecation error: {warning_msg}");
        } else {
            eprintln!("Deprecation warning: {warning_msg}");
        }
    }
}

/// Format a deprecation warning message
#[allow(dead_code)]
fn format_deprecation_warning(info: &DeprecationInfo) -> String {
    let mut msg = format!(
        "Function '{}' is deprecated since v{}",
        info.name, info.since
    );

    if let Some(remove_version) = &info.remove_in {
        msg.push_str(&format!(" and will be removed in v{remove_version}"));
    }

    msg.push_str(&format!(": {}", info.reason));

    if let Some(alternative) = &info.alternative {
        msg.push_str(&format!(" Use '{alternative}' instead."));
    }

    if let Some(notes) = &info.migration_notes {
        msg.push_str(&format!(" Migration notes: {notes}"));
    }

    msg
}

/// Macro to mark a function as deprecated with structured information
#[macro_export]
macro_rules! deprecated_function {
    (
        name = $name:expr,
        since = $since:expr,
        reason = $reason:expr
        $(, alternative = $alternative:expr)?
        $(, remove_in = $remove_in:expr)?
        $(, migration_notes = $notes:expr)?
        $(, level = $level:expr)?
    ) => {
        {
            use $crate::deprecation::{DeprecationInfo, DeprecationLevel, issue_deprecation_warning};

            let info = DeprecationInfo {
                name: $name.to_string(),
                since: $since.to_string(),
                remove_in: None $(.or(Some($remove_in.to_string())))?,
                level: DeprecationLevel::Soft $(.max($level))?,
                reason: $reason.to_string(),
                alternative: None $(.or(Some($alternative.to_string())))?,
                migration_notes: None $(.or(Some($notes.to_string())))?,
            };

            issue_deprecation_warning(&info);
        }
    };
}

/// Experimental feature marker
#[macro_export]
macro_rules! experimental_feature {
    ($name:expr, $description:expr) => {{
        use $crate::deprecation::{issue_deprecation_warning, DeprecationInfo, DeprecationLevel};

        let info = DeprecationInfo {
            name: $name.to_string(),
            since: "0.1.0-beta.1".to_string(),
            remove_in: None,
            level: DeprecationLevel::Experimental,
            reason: format!("This is an experimental feature: {}", $description),
            alternative: None,
            migration_notes: Some("API may change in future versions".to_string()),
        };

        issue_deprecation_warning(&info);
    }};
}

/// Registry of deprecated and experimental features in the library
pub struct FeatureRegistry;

impl FeatureRegistry {
    /// Get list of all deprecated features in the current version
    pub fn deprecated_features() -> Vec<DeprecationInfo> {
        vec![
            // GPU acceleration features (experimental)
            DeprecationInfo {
                name: "gpu_accelerated".to_string(),
                since: "0.1.0-alpha.1".to_string(),
                remove_in: None,
                level: DeprecationLevel::Experimental,
                reason: "GPU acceleration is experimental and may change".to_string(),
                alternative: None,
                migration_notes: Some(
                    "Enable with 'gpu' feature flag. API subject to change.".to_string(),
                ),
            },
            // Neural enhanced interpolation (experimental)
            DeprecationInfo {
                name: "neural_enhanced".to_string(),
                since: "0.1.0-alpha.3".to_string(),
                remove_in: None,
                level: DeprecationLevel::Experimental,
                reason: "Neural enhanced interpolation is experimental".to_string(),
                alternative: None,
                migration_notes: Some(
                    "API may change significantly in future versions".to_string(),
                ),
            },
            // Physics-informed interpolation (experimental)
            DeprecationInfo {
                name: "physics_informed".to_string(),
                since: "0.1.0-alpha.4".to_string(),
                remove_in: None,
                level: DeprecationLevel::Experimental,
                reason: "Physics-informed methods are experimental".to_string(),
                alternative: None,
                migration_notes: Some(
                    "Consider using standard RBF or spline methods for production".to_string(),
                ),
            },
            // Some Kriging variants with warnings
            DeprecationInfo {
                name: "experimental_kriging_variants".to_string(),
                since: "0.1.0-alpha.5".to_string(),
                remove_in: None,
                level: DeprecationLevel::Experimental,
                reason: "Some advanced Kriging variants show implementation warnings".to_string(),
                alternative: Some("enhanced_kriging::make_enhanced_kriging".to_string()),
                migration_notes: Some(
                    "Use stable Kriging implementations for production workloads".to_string(),
                ),
            },
        ]
    }

    /// Get features planned for removal in specific version
    pub fn features_removed_in(version: &str) -> Vec<DeprecationInfo> {
        Self::deprecated_features()
            .into_iter()
            .filter(|f| f.remove_in.as_ref().map(|v| v == version).unwrap_or(false))
            .collect()
    }

    /// Check if a feature is deprecated
    pub fn is_feature_deprecated(featurename: &str) -> bool {
        Self::deprecated_features()
            .iter()
            .any(|f| f.name == featurename)
    }

    /// Get deprecation info for a specific feature
    pub fn get_deprecation_info(featurename: &str) -> Option<DeprecationInfo> {
        Self::deprecated_features()
            .into_iter()
            .find(|f| f.name == featurename)
    }
}

/// Convenience functions for common deprecation scenarios
pub mod convenience {
    use crate::experimental_feature;

    /// Mark a GPU feature as experimental
    pub fn warn_gpu_experimental(featurename: &str) {
        experimental_feature!(
            featurename,
            "GPU acceleration support is experimental and may change significantly"
        );
    }

    /// Mark a neural network feature as experimental  
    pub fn warn_neural_experimental(featurename: &str) {
        experimental_feature!(
            featurename,
            "Neural network enhanced interpolation is experimental"
        );
    }

    /// Mark a physics-informed feature as experimental
    pub fn warn_physics_experimental(featurename: &str) {
        experimental_feature!(
            featurename,
            "Physics-informed interpolation methods are experimental"
        );
    }

    /// Issue a matrix conditioning warning
    pub fn warn_matrix_conditioning(condition_number: f64, context: &str) {
        if condition_number > 1e14 {
            eprintln!(
                "NUMERICAL WARNING: Poor matrix conditioning (condition number: {condition_number:.2e}) in {context}. \
                Consider regularization or data preprocessing."
            );
        }
    }

    /// Issue a performance warning for large datasets
    pub fn warn_performance_large_dataset(operation: &str, size: usize, threshold: usize) {
        if size > threshold {
            eprintln!(
                "PERFORMANCE WARNING: {operation} with {size} data points may be slow. \
                Consider using fast variants or GPU acceleration if available."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deprecation_config() {
        let config = DeprecationConfig::default();
        assert!(config.show_warnings);
        assert!(!config.warnings_as_errors);
        assert_eq!(config.max_warning_count, 3);
    }

    #[test]
    fn test_feature_registry() {
        let deprecated = FeatureRegistry::deprecated_features();
        assert!(!deprecated.is_empty());

        let gpu_deprecated = FeatureRegistry::is_feature_deprecated("gpu_accelerated");
        assert!(gpu_deprecated);

        let standard_feature = FeatureRegistry::is_feature_deprecated("spline");
        assert!(!standard_feature);
    }

    #[test]
    fn test_deprecation_info() {
        let info = DeprecationInfo {
            name: "test_function".to_string(),
            since: "0.1.0".to_string(),
            remove_in: Some("0.2.0".to_string()),
            level: DeprecationLevel::Soft,
            reason: "Test deprecation".to_string(),
            alternative: Some("new_function".to_string()),
            migration_notes: Some("Easy migration".to_string()),
        };

        let message = format_deprecation_warning(&info);
        assert!(message.contains("test_function"));
        assert!(message.contains("0.1.0"));
        assert!(message.contains("0.2.0"));
        assert!(message.contains("new_function"));
    }
}
