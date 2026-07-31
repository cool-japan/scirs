//! API compatibility checking for scirs2-core
//!
//! This module provides utilities to check API compatibility and ensure
//! that code using the library will work with specific versions.

use crate::apiversioning::{global_registry_mut, Version};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Check if a specific API is available in the current version
#[allow(dead_code)]
pub fn is_api_available(apiname: &str, module: &str) -> bool {
    let registry = global_registry_mut();
    let current_version = current_libraryversion();

    registry
        .apis_in_version(&current_version)
        .iter()
        .any(|entry| entry.name == apiname && entry.module == module)
}

/// Check if a set of APIs are all available
#[allow(dead_code)]
pub fn check_apis_available(apis: &[(&str, &str)]) -> CoreResult<()> {
    let mut missing = Vec::new();

    for (apiname, module) in apis {
        if !is_api_available(apiname, module) {
            missing.push(format!("{module}::{apiname}"));
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(CoreError::ValidationError(ErrorContext::new(format!(
            "Missing APIs: {}",
            missing.join(", ")
        ))))
    }
}

/// Get the current library version
#[allow(dead_code)]
pub fn current_libraryversion() -> Version {
    // Read version from Cargo.toml at compile time
    let versionstr = env!("CARGO_PKG_VERSION");
    Version::parse(versionstr).unwrap_or_else(|_| {
        // Fallback to hardcoded version if parsing fails
        Version::new(0, 1, 0)
    })
}

/// Check if the current version is compatible with a required version
#[allow(dead_code)]
pub fn is_version_compatible(required: &Version) -> bool {
    let current = current_libraryversion();
    current.is_compatible_with(required)
}

/// Macro to check API availability at compile time
///
/// Honestly a no-op today: it does not yet perform a real compile-time
/// API-existence check (that would require `$api`/`$module` to be actual
/// item paths rather than message fragments, plus a redesign of this
/// macro's public signature). It previously used a tautological
/// `assert!(true, ..)` here, which "passed" unconditionally regardless of
/// whether the referenced API existed — this version is an honest no-op
/// instead of masquerading as a check.
#[macro_export]
macro_rules! require_api {
    ($api:expr, $module:expr) => {
        const _: () = {
            let _ = concat!("API required: ", $module, "::", $api);
        };
    };
}

/// Runtime API compatibility checker
pub struct ApiCompatibilityChecker {
    required_apis: Vec<(String, String)>,
    minimum_version: Option<Version>,
}

impl Default for ApiCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiCompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        Self {
            required_apis: Vec::new(),
            minimum_version: None,
        }
    }

    /// Add a required API
    pub fn require_api(mut self, apiname: impl Into<String>, module: impl Into<String>) -> Self {
        self.required_apis.push((apiname.into(), module.into()));
        self
    }

    /// Set minimum version requirement
    pub fn minimum_version(mut self, version: Version) -> Self {
        self.minimum_version = Some(version);
        self
    }

    /// Check if all requirements are met
    pub fn check(&self) -> CoreResult<()> {
        // Check version compatibility
        if let Some(min_version) = &self.minimum_version {
            if !is_version_compatible(min_version) {
                return Err(CoreError::ValidationError(ErrorContext::new(format!(
                    "Version {} required, but current version is {}",
                    min_version,
                    current_libraryversion()
                ))));
            }
        }

        // Check API availability
        let apis: Vec<(&str, &str)> = self
            .required_apis
            .iter()
            .map(|(api, module)| (api.as_str(), module.as_str()))
            .collect();

        check_apis_available(&apis)
    }
}
